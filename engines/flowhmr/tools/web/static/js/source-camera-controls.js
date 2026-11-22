import * as api from './api.js';
import { cameraActive, cameraAvailable, cameraOperationMessage, cameraStatusForJob, cameraStatusMessage } from './camera-messages.js';
import { message as m, setAttr, setText } from './i18n.js';
import { $, downloadBlob, safeStem, show } from './utils.js';

export function bindSourceCamera({ showing, enter, exit, replaceCamera, exportTransform }) {
  const view = $('sourceCameraView'), retry = $('sourceCameraRetry'), cancel = $('sourceCameraCancel');
  const quickView = $('sourceCameraQuickView');
  const viewButtons = [view, quickView];
  const bar = $('sourceCameraBar'), notice = $('sourceCameraOperation');
  const exportMenu = $('sourceCameraExportMenu'), exportToggle = $('sourceCameraExport');
  let current = null, previewHint = '';

  function syncView() {
    const pressed = String(showing());
    for (const button of viewButtons) button.setAttribute('aria-pressed', pressed);
  }

  function render() {
    const { status, data, opening, preparing, cancelling, exporting, previewReady, error } = current ?? {};
    const cached = !!data || cameraAvailable(status);
    const active = cameraActive(status);
    show(bar, !!current);
    const viewTitle = cached && !previewReady ? m('view.loading')
      : cached ? previewHint || m('camera.preview') : m('camera.none');
    for (const button of viewButtons) {
      button.disabled = !cached || !previewReady || !!opening;
      setAttr(button, 'title', viewTitle);
    }
    syncView();
    show(quickView, cached);
    setText(retry, m(cached ? 'camera.recompute' : 'camera.generate'));
    retry.disabled = !current || !!preparing || active;
    exportToggle.setAttribute('aria-disabled', String(!cached || !!exporting));
    if (!cached || exporting) exportMenu.open = false;
    show(cancel, !!current && active);
    cancel.disabled = !!cancelling || status?.camera_state === 'cancel_requested';
    const operation = cameraOperationMessage(status) || (preparing && cached ? m('camera.recomputing_old') : '');
    const content = error || (exporting ? m('export.preparing') : operation || (opening ? m('camera.reading') :
      status ? cameraStatusMessage(status) : ''));
    setText(notice, content);
    setAttr(notice, 'title', status?.error ?? '');
    show(notice, !!content);
  }

  function receive(session, status) {
    if (current !== session) return;
    session.status = status;
    render();
  }

  function readCamera(session) {
    const revision = session.status.revision;
    if (session.data && (revision == null || revision === session.revision)) return Promise.resolve(session.data);
    if (session.read && session.read.revision === revision) return session.read.promise;
    const request = { revision };
    session.read = request;
    request.promise = api.getCamera(session.id, session.controller.signal).then((data) => {
      if (session !== current) return null;
      if (session.read !== request) return session.read?.promise ?? session.data;
      session.data = data;
      session.revision = data.revision ?? revision;
      session.error = '';
      replaceCamera(data);
      render();
      return data;
    }).catch((error) => {
      if (session !== current) return null;
      if (session.read !== request) return session.read?.promise ?? session.data;
      throw error;
    }).finally(() => {
      if (session.read === request) session.read = null;
    });
    return request.promise;
  }

  function refreshCamera(session) {
    if (session !== current || !session.data) return;
    return readCamera(session).catch((error) => {
      if (session !== current) return;
      session.error = error;
      render();
    });
  }

  function prepare(session, force) {
    return api.prepareCamera(session.id, session.controller.signal, (status) => receive(session, status), { force });
  }

  const toggleView = async () => {
    const session = current;
    if (!session || view.disabled) return;
    if (showing()) { exit(); return; }
    if (session.data) {
      enter();
      void refreshCamera(session);
      return;
    }
    session.opening = true;
    session.error = '';
    render();
    try {
      const data = await readCamera(session);
      if (data && current === session && session.previewReady) enter();
    } catch (error) {
      if (current === session && !error.cameraStatus) session.error = error;
    } finally {
      session.opening = false;
      if (current === session) render();
    }
  };
  for (const button of viewButtons) button.onclick = toggleView;

  retry.onclick = async () => {
    const session = current;
    if (!session || retry.disabled) return;
    session.preparing = true;
    session.error = '';
    render();
    try {
      await prepare(session, cameraAvailable(session.status));
      await refreshCamera(session);
    } catch (error) {
      if (current === session && !error.cameraStatus) session.error = error;
    } finally {
      session.preparing = false;
      if (current === session) render();
    }
  };

  exportToggle.onclick = (event) => {
    if (exportToggle.getAttribute('aria-disabled') === 'true') event.preventDefault();
  };
  $('sourceCameraFormats').onclick = async (event) => {
    const format = event.target.closest('[data-camera-format]')?.dataset.cameraFormat;
    const session = current;
    if (!format || !session || session.exporting || !cameraAvailable(session.status)) return;
    exportMenu.open = false;
    session.exporting = true;
    session.error = '';
    render();
    try {
      const signal = session.controller.signal;
      const transform = await exportTransform?.(signal);
      const content = await api.exportCamera(session.id, format, transform, signal);
      if (current === session) {
        downloadBlob(new Blob([content], { type: 'application/zip' }), `${safeStem(session.filename)}_camera_${format}.zip`);
      }
    } catch (error) {
      if (current === session) session.error = error;
    } finally {
      session.exporting = false;
      if (current === session) render();
    }
  };

  cancel.onclick = async () => {
    const session = current;
    if (!session || cancel.disabled || !cameraActive(session.status)) return;
    session.cancelling = true;
    session.error = '';
    render();
    try {
      const status = await api.cancelCamera(session.id, session.controller.signal);
      if (cameraActive(session.status) || !cameraActive(status)) receive(session, status);
    } catch (error) {
      if (current === session) session.error = error;
    } finally {
      session.cancelling = false;
      if (current === session) render();
    }
  };

  render();
  return {
    syncView,
    select(id, job = {}, previewReady = true) {
      current?.controller.abort();
      current = id ? { id, filename: job.filename, status: cameraStatusForJob(job), previewReady,
        controller: new AbortController() } : null;
      exportMenu.open = false;
      render();
    },
    update(job) {
      const session = current;
      if (!session || session.id !== job.id || session.preparing || session.opening || session.cancelling) return;
      const status = cameraStatusForJob(job);
      if (status.revision != null && status.revision !== session.status.revision) session.error = '';
      receive(session, status);
      void refreshCamera(session);
    },
    preparePreview() {
      if (current) current.previewReady = true;
      render();
    },
    setPreviewHint(hint) {
      previewHint = hint;
      render();
    },
  };
}
