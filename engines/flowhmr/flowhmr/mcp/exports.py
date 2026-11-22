"""Persist export receipts and publish complete downloads atomically."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import uuid
from contextlib import nullcontext
from zipfile import ZipFile, ZIP_DEFLATED, is_zipfile
import shutil
from datetime import datetime, timezone
from .backend import fail, local_file


class Exports:
    def __init__(self, directory, backend):
        self.directory = Path(directory)
        self.backend = backend
        self.rows = {}
        self.tasks = set()

    def save(self, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (row['export_id'] + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
        self.rows[row['export_id']] = row

    def recover(self):
        if not self.directory.exists():
            return
        for path in self.directory.glob('*.json'):
            row = json.loads(path.read_text('utf-8'))
            if row['status'] in ('queued', 'running'):
                # A complete published file is authoritative only with a saved done receipt.
                row.update(status='failed', error={'code': 'export_interrupted',
                    'message': 'MCP restarted before a completion receipt; inspect the target before retrying.'})
                temporary = row.get('temporary_path')
                if temporary:
                    Path(temporary).unlink(missing_ok=True)
                self.save(row)
            self.rows[row['export_id']] = row

    async def close(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    async def notify(self):
        await self.backend.request('POST', '/api/mcp/events', json={'kind': 'exports'})

    def get(self, export_id):
        if export_id not in self.rows:
            fail('export_not_found', 'Unknown export ID.')
        return {k: v for k, v in self.rows[export_id].items() if k != 'temporary_path'}

    async def submit(self, options):
        job = await self.backend.job(options.job_id)
        if job['status'] != 'done':
            fail('scene_not_ready' if job.get('scene_needs_build') else 'job_not_ready',
                 'Generate the motion and prepare its scene before exporting.')
        people = options.person_ids or job['person_ids']
        if options.person_ids is not None and not set(people) <= set(job['person_ids']):
            fail('invalid_people', 'Use person_ids from flowhmr_get_job.')
        if options.format == 'fbx' and not job.get('has_fbx'):
            fail('export_unavailable', 'FBX export is unavailable for this result.')
        if options.format == 'fbx' and (options.template_path or options.fbx_profile == 'maya_adv') and len(people) != 1:
            fail('invalid_people', 'This profile requires exactly one person.')
        extension = 'zip' if options.include_camera or (options.format in ('vmd', 'vrma') and len(people) > 1) else options.format
        path = Path(options.output_path)
        if not path.is_absolute() or str(path).startswith(('\\\\.\\', '\\\\?\\')):
            fail('invalid_path', 'output_path must be an absolute ordinary file path.')
        if not path.parent.is_dir() or path.suffix.lower() != '.' + extension:
            fail('invalid_path', f'Target parent must exist and extension must be .{extension}.')
        if path.is_symlink() or (path.exists() and not path.is_file()):
            fail('invalid_path', 'Target must be a regular file, not a directory or symlink.')
        path = path.parent.resolve() / path.name
        if path.exists() and not options.overwrite:
            fail('output_exists', 'Target exists. Choose another path or explicitly enable overwrite.')
        template = local_file(options.template_path, {'.fbx'}) if options.template_path else None
        if template and template.stat().st_size > 128 * 1024 * 1024:
            fail('template_too_large', 'FBX templates must not exceed 128 MiB.')
        row = dict(export_id=uuid.uuid4().hex, job_id=options.job_id, status='queued',
                   path=str(path), format=options.format, container=extension, person_ids=people,
                   created=datetime.now(timezone.utc).isoformat(), error=None,include_camera=options.include_camera)
        self.save(row)
        await self.notify()
        task = asyncio.create_task(self.run(row, options, template))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return self.get(row['export_id'])

    async def prepare_camera(self,row):
        status=await self.backend.request('POST',f"/api/jobs/{row['job_id']}/camera/prepare",json={})
        deadline=asyncio.get_running_loop().time()+3600
        while not status.get('available', status['camera_state'] == 'ready'):
            if status['camera_state'] in ('failed','cancelled','none'):
                fail('camera_'+status['camera_state'],status.get('error') or status.get('message') or 'Camera is unavailable.')
            if asyncio.get_running_loop().time()>deadline:
                fail('camera_wait_timeout','Camera is still running. Inspect its status before retrying export.')
            row['phase']='waiting_for_camera';row['camera_state']=status['camera_state'];self.save(row)
            await asyncio.sleep(.7)
            status=await self.backend.request('GET',f"/api/jobs/{row['job_id']}/camera/status")
        row['phase']='exporting';row['camera_state']='ready';self.save(row)

    async def package(self,motion_file,options):
        """Reuse the same motion/camera HTTP exporters as the GUI; never regenerate motion."""
        with tempfile.TemporaryDirectory(prefix='.flowhmr-pair-',dir=Path(motion_file).parent) as temp:
            root=Path(temp);paired=root/'result.zip'
            def copy_members(output,source,prefix,raw_name):
                if not raw_name.endswith('.npz') and is_zipfile(source):
                    with ZipFile(source) as archive:
                        for info in archive.infolist():
                            if info.is_dir():continue
                            name=info.filename.replace('\\','/')
                            if name.startswith('/') or '..' in name.split('/') or ':' in name:
                                raise ValueError('Invalid file name in export archive')
                            with archive.open(info) as src,output.open(prefix+name,'w') as dst:shutil.copyfileobj(src,dst)
                else:output.write(source,prefix+raw_name)
            with ZipFile(paired,'w',ZIP_DEFLATED) as output:
                copy_members(output,motion_file,'motion/' if options.include_camera else '',f'motion.{options.format}')
                if options.include_camera:
                    camera_path=root/'camera.zip'
                    async with self.backend.client(timeout=1200) as client:
                        async with client.stream('POST',f'/api/jobs/{options.job_id}/camera/export',json={'format':options.format}) as response:
                            await self.backend.check(response)
                            with camera_path.open('wb') as dst:
                                async for block in response.aiter_bytes():dst.write(block)
                    if not is_zipfile(camera_path):raise ValueError('Invalid camera export archive')
                    copy_members(output,camera_path,'camera/','camera.zip')
                    output.writestr('pairing.json',json.dumps(dict(job_id=options.job_id,format=options.format,
                        world='paired_motion_y_up_metres',camera_transform=dict(scale=1,translation=[0,0,0]),
                        target='source body; selected preview character offsets are not baked',
                        motion_directory='motion',camera_directory='camera'),ensure_ascii=False,indent=2))
                    output.writestr('READ_ME.txt','Motion and camera share source frame numbers and the source-body world.\n'
                        'Load motion/ and camera/ together. See camera/export-report.json for format approximations.\n'
                        'For a selected PMX character and its matching camera, use the desktop export dialog.\n')
            os.replace(paired,motion_file)

    async def run(self, row, options, template):
        temporary = None
        try:
            row['status'] = 'running'
            path = Path(row['path'])
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.flowhmr-export-', suffix='.tmp')
            os.close(descriptor)
            row['temporary_path'] = temporary
            self.save(row)
            await self.notify()
            if options.include_camera:
                await self.prepare_camera(row)
            params = []
            if options.person_ids:
                params.extend(('person', str(person)) for person in options.person_ids)
            if options.format == 'fbx':
                params.extend([('profile', options.fbx_profile), ('mesh', '1' if options.include_mesh else '0'),
                               ('first_pose', options.reference_pose)])
            with template.open('rb') if template else nullcontext() as file:
                async with self.backend.client(timeout=1200) as client:
                    async with client.stream('POST' if template else 'GET',
                        f'/api/jobs/{options.job_id}/{options.format}', params=params,
                        files={'template': (template.name, file, 'application/octet-stream')} if template else None) as response:
                        await self.backend.check(response)
                        size = 0
                        with open(temporary, 'wb') as output:
                            async for block in response.aiter_bytes():
                                output.write(block)
                                size += len(block)
                        if not size:
                            raise ValueError('Backend returned an empty export')
            if options.include_camera:
                await self.package(temporary,options)
                size=Path(temporary).stat().st_size
            if options.overwrite:
                os.replace(temporary, path)
            elif os.name == 'nt':
                os.rename(temporary, path)  # Windows rename refuses an existing target.
            else:
                os.link(temporary, path)
                os.unlink(temporary)
            temporary = None
            row.update(status='done', size_bytes=size, finished=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            try:
                detail = json.loads(str(error))
            except (ValueError, TypeError):
                detail = {'code': 'output_exists' if isinstance(error, FileExistsError) else 'export_failed',
                          'message': str(error)}
            row.update(status='failed', error=detail)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
            row.pop('temporary_path', None)
            self.save(row)
            await self.notify()
