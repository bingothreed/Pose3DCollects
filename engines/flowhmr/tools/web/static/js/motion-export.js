import { $, downloadBlob, downloadFile, el, personColor, safeStem, show } from "./utils.js";
import { appError, message as m, setText } from "./i18n.js";
import * as api from "./api.js";
import { Vector3 } from 'three';
import { animatedCameraTransform, pelvisCameraTransform } from './paired-camera.js';
import { scenePlacement } from './frame-sample.js';
import { cameraAvailable, cameraStatusMessage, cameraStatusForJob } from './camera-messages.js';
import { unzipSync, zipSync } from '../vendor/three/fbx/libs/fflate.module.js';

const FORMAT_STORAGE = "flowhmr-export-format";
const SKIN_STORAGE = "flowhmr-fbx-with-skin";
const POSE_STORAGE = "flowhmr-fbx-first-pose";
const CAMERA_FORMATS = ['flowhmr', 'vmd', 'npz'];

export function initMotionExport(getBodyVariant, viewer) {
  const dialog = $("fbxExportDialog"), button = $("dlFbx"), menu = $("menuFbx");
  const format = $("motionExportFormat"), confirm = $("fbxConfirm");
  const skin = $("fbxWithSkin"), pose = $("fbxFirstPose"), peopleField = $("fbxPeople");
  const error = $("motionExportError"), openCharacters = $("fbxChooseCharacter");
  const mixamoRig = $("mixamoRig"), mixamoCharacter = $("mixamoRigCharacter");
  const vmdTarget = $("vmdTarget"), characterOption = $("vmdTargetCharacter");
  const chooseCharacter = $("vmdChooseCharacter");
  const includeCamera = $('includeCamera');
  const formats = [...format.options];
  let preferred = localStorage.getItem(FORMAT_STORAGE);
  if (preferred === "mixamo_iclone") preferred = "mixamo";
  let result = null, pending = null, entries = [], multiPeople = [], single = false;
  let pairedExport = false;

  skin.checked = localStorage.getItem(SKIN_STORAGE) === "on";
  const savedPose = localStorage.getItem(POSE_STORAGE);
  pose.value = ["t", "a"].includes(savedPose) ? savedPose : "none";

  const selected = () => entries.filter(({ input }) => input.checked).map(({ input }) => input.value);
  const targets = () => viewer.getPmxActors(selected());
  const retargeted = () => viewer.getRetargetActors(selected());
  const fbxActor = (profile) => viewer.getFbxActor(selected(), profile);
  function selectMode(next) {
    if (next === single) return;
    if (next) multiPeople = selected();
    const keep = next ? [multiPeople[0] ?? entries[0]?.input.value] : multiPeople;
    for (const { input } of entries) {
      input.type = next ? "radio" : "checkbox";
      if (next) input.name = "fbx-export-person"; else input.removeAttribute("name");
      input.checked = keep.includes(input.value);
    }
    single = next;
  }
  function showError(value = "") {
    setText(error, value);
    show(error, !!value);
  }
  function sync() {
    const glb = formats.find(({value})=>value==="retarget_glb");
    glb.disabled = !result || !retargeted().length;
    for (const option of formats) {
      option.hidden = (pairedExport && !CAMERA_FORMATS.includes(option.value)) || (option === glb && glb.disabled);
    }
    if (glb.disabled && format.value==="retarget_glb") format.value="flowhmr";
    const profile = format.value;
    const mixamo = profile === "mixamo", metahuman = profile === "metahuman";
    const currentFbx = profile === "retarget_fbx";
    const singlePerson = currentFbx || metahuman || profile === "maya_adv";
    selectMode(singlePerson);
    const actor = mixamo || metahuman || currentFbx ? fbxActor(profile) : null;
    const useActor = actor && (!mixamo || mixamoRig.value !== "generic");
    show("mixamoRigOptions", mixamo && !!actor);
    if (mixamo && actor) setText(mixamoCharacter, m("export.fbx_target", { name: actor.asset.name }));
    show(openCharacters, (metahuman || currentFbx) && !actor);
    show("standardFbxOptions", profile === "flowhmr" || mixamo || (currentFbx && !!actor));
    show("fbxReferenceOptions", profile === "flowhmr" || (mixamo && !useActor));
    setText($("fbxSkinHint"), m(useActor ? "export.fbx_target_skin" : "export.skin_hint"));
    show("vmdOptions", profile === "vmd");
    const roles = targets();
    const cameraSupported = CAMERA_FORMATS.includes(profile);
    show('pairedCameraOptions', cameraSupported);
    includeCamera.disabled = !!pending || !result;
    if (!cameraSupported) includeCamera.checked = false;
    if (!pending) setText($('pairedCameraHint'), m(result?.cameraAvailable ? 'export.camera_ready' : 'export.camera_missing'));
    characterOption.disabled = characterOption.hidden = !roles.length;
    if (roles.length) setText(characterOption, roles.length === 1
      ? m("export.vmd_character", { name: roles[0].asset.name }) : m("export.vmd_characters"));
    else vmdTarget.value = "generic";
    let hint = profile;
    if (profile === "vmd") hint = vmdTarget.value === "target" ? "vmd_target" : "vmd_generic";
    if (mixamo) hint = useActor ? "mixamo_target" : "mixamo";
    if (metahuman && !actor) hint = "metahuman_required";
    if (currentFbx && !actor) hint = "retarget_fbx_required";
    setText($("motionExportHint"), profile === "retarget_glb" ? m("retarget.glb_hint") :
      hint ? m(`export.${hint}_hint`, { name: actor?.asset.name }) : "");
    setText($("exportPeopleLabel"), m(singlePerson ? "export.select_person" : "export.select_people"));
    const busy = !!pending;
    format.disabled = peopleField.disabled = vmdTarget.disabled = chooseCharacter.disabled =
      mixamoRig.disabled = openCharacters.disabled = skin.disabled = busy;
    confirm.disabled = !result || !profile || busy || (pairedExport && !includeCamera.checked) || (entries.length > 0 && !selected().length)
      || ((metahuman || currentFbx) && !actor);
    setText(confirm, m(busy ? "export.preparing" : "common.export"));
  }
  function setPeople(scene) {
    single = false;
    entries = (scene?.people ?? []).map(({ person }) => {
      const label = el("label", "fbx-export-person");
      const input = el("input");
      const dot = el("span", "person-dot");
      dot.style.background = personColor(person - 1);
      input.type = "checkbox";
      input.value = person;
      input.checked = true;
      label.append(input, dot, el("span", "", m("person.number", { number: person })));
      return { label, input };
    });
    multiPeople = selected();
    $("fbxPeopleList").replaceChildren(...entries.map(({ label }) => label));
    show(peopleField, !!entries.length);
  }
  function setButtons(enabled) {
    button.disabled = menu.disabled = !enabled;
    show(button, enabled);
  }
  function saveExport(blob, filename) {
    dialog.close("export");
    downloadBlob(blob, filename);
  }

  function open(withCamera = false) {
    if (!result || dialog.open) return;
    if (pairedExport) includeCamera.checked = false;
    pairedExport = withCamera;
    dialog.returnValue = "";
    showError();
    vmdTarget.value = targets().length ? "target" : "generic";
    mixamoRig.value = "target";
    // Resolve the default after preview and character restoration have run.
    const currentFbx = formats.find(({value}) => value === "retarget_fbx");
    if (withCamera) {
      const paired = formats.filter((option) => !option.disabled && CAMERA_FORMATS.includes(option.value));
      format.value = (paired.find((option) => option.value === preferred) ?? paired[0])?.value ?? "";
      includeCamera.checked = true;
    } else if ((!preferred || preferred === "retarget_glb") && !currentFbx.disabled && fbxActor("retarget_fbx")) {
      format.value = "retarget_fbx";
    } else if (preferred === "retarget_glb" && retargeted().length) format.value = preferred;
    else if (preferred && formats.some((option) => !option.disabled && option.value === preferred)) format.value = preferred;
    sync();
    dialog.showModal();
  }
  button.onclick = () => open();
  $('sourceCameraExportPair').onclick = () => open(true);
  menu.onclick = () => button.click();
  peopleField.addEventListener("change", sync);
  includeCamera.addEventListener('change', sync);
  vmdTarget.addEventListener("change", sync);
  mixamoRig.addEventListener("change", () => { showError(); sync(); });
  openCharacters.onclick = () => {
    dialog.close("character");
    viewer.openCharacter(selected()[0]);
  };
  chooseCharacter.onclick = () => {
    dialog.close("character");
    viewer.openCharacter();
  };
  window.addEventListener("flowhmr:characterchange", () => {
    if (!dialog.open || pending) return;
    vmdTarget.value = targets().length ? "target" : "generic";
    sync();
  });
  format.addEventListener("change", () => {
    showError();
    sync();
  });
  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    pending?.abort();
    pending = null;
    sync();
  });
  dialog.querySelector("form").addEventListener("submit", async (event) => {
    if (event.submitter !== confirm) return;
    event.preventDefault();
    if (!result || confirm.disabled || pending) return;
    const { id } = result;
    const profile = format.value;
    const people = selected();
    preferred = profile;
    localStorage.setItem(FORMAT_STORAGE, profile);
    const targetVmd = profile === "vmd" && vmdTarget.value === "target";
    const vmdActors = profile === "vmd" ? viewer.getVmdActors(people) : [];
    const correctedVmd = !targetVmd && vmdActors.some((member) => member.poseOffsets);
    const actor = profile === "retarget_fbx" || profile === "metahuman" || (profile === "mixamo" && mixamoRig.value !== "generic")
      ? fbxActor(profile) : null;
    if ((profile === "metahuman" || profile === "retarget_fbx") && !actor) { sync(); return; }
    if (includeCamera.checked) {
      const operation = pending = new AbortController();
      showError();
      // Preserve the user's selection while marking the operation busy.
      sync();
      try {
        await api.prepareCamera(id, operation.signal, (status) => {
          if (dialog.open) setText($('pairedCameraHint'), cameraStatusMessage(status));
        });
        result.cameraAvailable = true;
        let files, transform = { scale: 1, translation: [0, 0, 0] };
        if (targetVmd || correctedVmd) {
          const target = await makeVmd(vmdActors, result.filename, targetVmd, operation.signal, { withCamera: true });
          files = target.files;
          transform = target.cameraTransform ?? transform;
        } else {
          const url = profile === 'npz' ? api.npzUrl(id) : profile === 'vmd' ? api.motionUrl(id, 'vmd', people)
            : api.fbxUrl(id, skin.checked, getBodyVariant(), people, pose.value, 'flowhmr');
          const motion = await api.getMotionDownload(url, operation.signal);
          files = motion.type?.startsWith('application/zip')
            ? Object.entries(unzipSync(motion.bytes)).map(([name, bytes]) => ({ name, bytes }))
            : [{ name: `motion.${profile === 'flowhmr' ? 'fbx' : profile}`, bytes: motion.bytes }];
        }
        const cameraZip = new Uint8Array(await api.exportCamera(id, profile === 'flowhmr' ? 'fbx' : profile, transform, operation.signal));
        files.push(...Object.entries(unzipSync(cameraZip)).map(([name, bytes]) => ({ name, bytes })));
        if (pending !== operation || result?.id !== id || !dialog.open) return;
        const stem = safeStem(result.filename);
        const entries = Object.create(null);
        for (const { name, bytes } of files) {
          if (!name || /[/\\\x00-\x1f]/.test(name) || Object.hasOwn(entries, name)) throw new Error('Invalid or duplicate paired export filename');
          entries[name] = bytes;
        }
        saveExport(new Blob([zipSync(entries, { level: 1 })], { type: 'application/zip' }), `${stem}_motion_camera.zip`);
      } catch (cause) {
        if (!operation.signal.aborted && dialog.open) showError(cause);
      } finally {
        if (pending === operation) { pending = null; sync(); }
      }
      return;
    }
    if (profile === 'npz') {
      dialog.close('export');
      downloadFile(api.npzUrl(id));
      return;
    }
    if (profile === "vrma" || (profile === "vmd" && !targetVmd && !correctedVmd)) {
      dialog.close("export");
      downloadFile(api.motionUrl(id, profile, people));
      return;
    }
    if (!actor && profile !== "vmd" && profile !== "retarget_glb") {
      const advancedSkeleton = profile === "maya_adv";
      if (!advancedSkeleton) {
        localStorage.setItem(SKIN_STORAGE, skin.checked ? "on" : "off");
        localStorage.setItem(POSE_STORAGE, pose.value);
      }
      dialog.close("export");
      downloadFile(api.fbxUrl(id, !advancedSkeleton && skin.checked, getBodyVariant(), people,
        advancedSkeleton ? "t" : pose.value, profile));
      return;
    }
    const operation = pending = new AbortController();
    showError();
    sync();
    try {
      let exported;
      if (profile==="retarget_glb") {
        const { bakeRetargetGlb }=await import("./retarget-export.js");
        const stem=(result.filename||"FlowHMR").replace(/\.[^.]+$/,"").replace(/[<>:"/\\|?*\x00-\x1f]/g,"_").slice(0,100);
        exported=await bakeRetargetGlb(retargeted(),stem,operation.signal);
      } else if (profile === "vmd") {
        exported = await makeVmd(vmdActors, result.filename, targetVmd, operation.signal);
      } else {
        const { bakeTargetFbx } = await import("./retarget-fbx.js");
        const baked = await bakeTargetFbx(actor, operation.signal, profile);
        const offersSkin = profile === "mixamo" || profile === "retarget_fbx";
        const withSkin = offersSkin && skin.checked;
        if (offersSkin) localStorage.setItem(SKIN_STORAGE, withSkin ? "on" : "off");
        exported = await api.exportTargetFbx(id, people, profile, actor, baked, withSkin, operation.signal);
      }
      if (pending !== operation || result?.id !== id || !dialog.open) return;
      saveExport(exported.blob, exported.filename);
    } catch (cause) {
      if (!operation.signal.aborted && dialog.open) {
        showError(profile === "vmd" ? m("export.pmx_vmd_failed", { detail: cause }) : cause);
      }
    } finally {
      if (pending === operation) {
        pending = null;
        sync();
      }
    }
  });

  return {
    setResult(id, job) {
      result = { id, filename: job.filename, cameraAvailable: cameraAvailable(cameraStatusForJob(job)) };
      includeCamera.checked = false;
      const availableByFormat = { vmd: !!job.has_vmd, vrma: !!job.has_vrma, npz: !!job.has_vmd };
      for (const option of formats) option.disabled = !(availableByFormat[option.value] ?? job.has_fbx);
      const available = formats.filter((option) => !option.disabled && option.value!=="retarget_glb");
      format.value = (available.find((option) => option.value === preferred) ?? available[0])?.value ?? "";
      setPeople(job.scene);
      setButtons(!!available.length);
      sync();
    },
    updateCameraStatus(job) {
      if (result?.id !== job.id) return;
      const available = cameraAvailable(cameraStatusForJob(job));
      if (result.cameraAvailable === available) return;
      result.cameraAvailable = available;
      sync();
    },
    clear() {
      pending?.abort();
      pending = null;
      result = null;
      if (dialog.open) dialog.close("cancel");
      setPeople(null);
      setButtons(false);
      sync();
    },
  };
}

export async function makeVmd(actors, filename, target, signal, { withCamera = false } = {}) {
  if (!actors.length) throw appError("export.vmd_character_unavailable");
  const [solverModule, { encodePmxVmd, packPmxVmdZip }] = await Promise.all([
    target ? import("./pmx-retarget.js") : import("./generic-vmd.js"), import("./pmx-vmd.js"),
  ]);
  const stem = safeStem(filename);
  const files = [];
  let cameraTransform = null;
  for (const actor of actors) {
    signal.throwIfAborted();
    const { asset } = actor;
    const motion = actor.motion ?? await api.getCharacterMotion(actor.id, signal);
    const clone = target ? asset.clone() : null;
    let solver;
    try {
      solver = target ? solverModule.createPmxRetarget(clone, asset.rest, motion, { poseOffsets: actor.poseOffsets })
        : solverModule.genericVmdRig(motion, actor.poseOffsets);
      const rig = solver.cameraRig;
      const sceneTrack = actor.sceneTrack ?? { yaw: actor.yaw, offset: actor.offset };
      const placement = frame => scenePlacement(sceneTrack, actor.start + frame);
      let onSample;
      if (withCamera && !files.length && rig) {
        const samples = [], pelvis = new Vector3();
        cameraTransform = animatedCameraTransform(rig.scale, actor.cameraStart, samples);
        onSample = (frame) => {
          rig.sampledPelvis(pelvis);
          samples.push(pelvisCameraTransform(rig.scale, motion, frame, pelvis.toArray(), placement(frame)).translation);
        };
      }
      const bytes = await encodePmxVmd({
        solver, rest: target ? asset.rest : solver.rest, modelName: target ? asset.name : "FlowHMR",
        frameCount: motion.frames, fps: actor.fps, startFrame: actor.start, sceneEndFrame: actor.sceneEnd,
        placement, visibleRanges:sceneTrack.visible_ranges, signal, onSample,
      });
      const tag = actor.person == null ? "" : `_person${actor.person}`;
      files.push({ name: `${stem}${tag}_${target ? "pmx" : "vmd"}.vmd`, bytes });
    } finally {
      if (clone) asset.releaseClone(clone);
      else solver?.dispose();
    }
  }
  return { files, cameraTransform, ...(files.length === 1
    ? { blob: new Blob([files[0].bytes], { type: "application/octet-stream" }), filename: files[0].name }
    : { blob: packPmxVmdZip(files), filename: `${stem}${target ? "_pmx" : ""}_vmd.zip` }) };
}
