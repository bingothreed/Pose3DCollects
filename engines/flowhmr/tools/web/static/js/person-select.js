// Person selection card.
//  - choosing: one row per detected person (thumbnail, colour matching its box in the video
//    preview, frame count); toggle any rows or video boxes, then confirm together.
//  - locked:   each queued or finished job shows its own selected person.
import { $, show, el, personColor } from "./utils.js";
import { message as m, setAttr, setText } from "./i18n.js";
import * as api from "./api.js";

const CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

// Tracking fragments merged into this person by the crossing optimisation.
function mergedTrackingIds(source) {
  const ids = source?.bytetrack_ids || [];
  const own = source?.track_id;
  return ids.filter((id) => id !== own);
}

function appendMergedIds(container, source) {
  const merged = mergedTrackingIds(source);
  if (!merged.length) return;
  container.append(el("span", "person-meta num", m("person.merged_ids", { ids: merged.join(", ") })));
}

export function initPersonSelect(bus) {
  let jobId = null;
  let candidates = [];
  let defaultIdx = null;
  let previewIndex = null, lockedIndex = null;
  const selected = new Set();
  const groups = [];
  let rows = [];
  let mode = "hidden";      // "hidden" | "choosing" | "locked"
  let submitting = false, confirmed = false;
  let suggested = null;

  const list = $("personList");
  const joinTracks = $("personJoinTracks");
  const groupList = $("personGroups");
  const groupNext = $("personGroupNext");
  const confirmButton = $("personConfirm");
  joinTracks.addEventListener("change", () => {
    if (!joinTracks.checked) {
      for (const group of groups) group.forEach(index => selected.add(index));
      groups.length = 0;
    }
    setError("");
    syncChoosing();
    emitSelection();
  });

  function emitSelection() {
    bus.emit("person:selection", { selected: [...groups.flat(), ...selected], previewIndex });
  }

  function thumb(i, cls) {
    const box = el("span", cls);
    box.style.borderColor = personColor(i);
    const img = el("img");
    setAttr(img, "alt", m("person.number", { number: i + 1 }));
    img.loading = "lazy";
    img.src = api.personThumbUrl(jobId, i);
    img.onerror = () => { img.remove(); box.classList.add("no-img"); box.textContent = `${i + 1}`; };
    box.append(img);
    return box;
  }

  function renderChoosing() {
    list.innerHTML = "";
    rows = candidates.map((c, i) => {
      const btn = el("button", "person");
      btn.type = "button";
      btn.append(thumb(i, "person-thumb"));
      const text = el("span", "person-text");
      const name = el("span", "person-name");
      const chip = el("span", "person-dot");
      const tag = el("span", "person-tag");
      chip.style.background = personColor(i);
      name.append(chip, el("span", "", m("person.number", { number: i + 1 })), tag);
      text.append(name, el("span", "person-meta num",
        m("person.track_duration", { frames: c.num_frames, seconds: (c.num_frames / 30).toFixed(1), start: c.start + 1, end: c.end + 1 })));
      appendMergedIds(text, c);
      btn.append(text);
      const mark = el("span", "person-check");
      mark.innerHTML = CHECK_SVG;
      btn.append(mark);
      btn.addEventListener("click", () => togglePerson(i));
      list.append(btn);
      return { btn, tag };
    });
    syncChoosing();
  }

  function syncChoosing() {
    const busy = submitting || confirmed;
    const joining = joinTracks.checked;
    const assignment = new Map();
    groups.forEach((group, number) => group.forEach(index => assignment.set(index, number + 1)));
    joinTracks.disabled = busy;
    show("personJoinRow", candidates.length > 1);
    show("personJoinHelp", joining);
    show(groupList, groups.length > 0);
    groupList.replaceChildren(...groups.map((group, index) => {
      const button = el("button", "btn-secondary");
      button.type = "button";
      button.disabled = busy;
      setText(button, m("person.group_chip", { number: index + 1, count: group.length }));
      setAttr(button, "aria-label", m("person.group_undo", { number: index + 1 }));
      button.addEventListener("click", () => {
        groups.splice(index, 1);
        if (!selected.size) group.forEach(fragment => selected.add(fragment));
        setError("");
        syncChoosing();
        emitSelection();
      });
      return button;
    }));
    show(groupNext, joining);
    groupNext.disabled = busy || !selected.size;
    setText(groupNext, m("person.group_add", { number: groups.length + 1 }));
    rows.forEach(({ btn, tag }, i) => {
      const checked = selected.has(i) || assignment.has(i);
      btn.classList.toggle("selected", checked);
      btn.setAttribute("aria-pressed", String(checked));
      btn.disabled = busy;
      setText(tag, assignment.has(i) ? m("person.group_assigned", { number: assignment.get(i) })
        : i === previewIndex ? m("person.previewing") : i === defaultIdx ? m("common.recommended") : "");
      show(tag, !!tag.textContent);
    });
    const acceptsSuggestion = joining && !groups.length && suggested?.length === selected.size
      && suggested.every(i => selected.has(i));
    let countText = m("person.selection_help", { detected: candidates.length, selected: selected.size });
    if (acceptsSuggestion) countText = m("person.join_suggestion", { count: selected.size });
    else if (joining) {
      countText = assignment.size === candidates.length
        ? m("person.group_ready", { count: groups.length })
        : m("person.group_step", { number: groups.length + 1, count: selected.size });
    }
    setText($("personCount"), countText);
    confirmButton.disabled = busy || (!selected.size && !groups.length);
    setText(confirmButton, submitting ? m("job.submitting") : confirmed ? m("person.confirmed")
      : groups.length ? m("person.group_generate", { count: groups.length + Number(selected.size > 0) })
      : !selected.size ? m("person.check") : selected.size === 1 ? m("person.confirm_one", { number: selected.values().next().value + 1 })
      : m(joining ? "person.confirm_join" : "person.confirm_multiple", { count: selected.size }));
  }

  function renderLocked(job, candidate) {
    list.innerHTML = "";
    const row = el("div", "person locked");
    row.append(thumb(lockedIndex, "person-thumb large"));
    const text = el("span", "person-text");
    const name = el("span", "person-name");
    const chip = el("span", "person-dot");
    chip.style.background = "#10b981";
    name.append(chip, el("span", "", m("person.number", { number: lockedIndex + 1 })));
    const how = m(job.track_fragments?.length ? "person.joined" : job.selection_auto ? "person.chosen_auto" : "person.chosen_manual");
    text.append(name, el("span", "person-meta", how));
    if (candidate) text.append(el("span", "person-meta num",
      m("person.track", { frames: candidate.num_frames, start: candidate.start + 1, end: candidate.end + 1 })));
    appendMergedIds(text, candidate ?? job.person_track);
    row.append(text);
    list.append(row);
  }

  function togglePerson(i) {
    if (mode !== "choosing" || submitting || confirmed || !candidates[i]) return;
    if (!groups.some(group => group.includes(i))) {
      if (selected.has(i)) selected.delete(i); else selected.add(i);
    }
    setError("");
    previewIndex = i;
    syncChoosing();
    emitSelection();
  }

  async function submit(addPerson) {
    if (!jobId || mode !== "choosing" || submitting || confirmed) return;
    if (!selected.size && (addPerson || !groups.length)) return;
    const id = jobId;
    const draft = [...selected];
    submitting = true;
    setError("");
    syncChoosing();
    try {
      const result = groups.length || addPerson
        ? await api.selectPersonGroups(id, draft.length ? [...groups, draft] : groups, addPerson)
        : await api.selectPeople(id, draft, joinTracks.checked && draft.length > 1);
      if (jobId !== id || mode !== "choosing") return;
      if (addPerson) {
        groups.push(draft);
        selected.clear();
        emitSelection();
      } else {
        confirmed = true;
        bus.emit("person:confirmed", result.people);
      }
    } catch (e) {
      if (jobId === id && mode === "choosing") setError(e.message ? e : m("person.choose_failed"));
    } finally {
      if (jobId === id) {
        submitting = false;
        if (mode === "choosing") syncChoosing();
      }
    }
  }

  groupNext.addEventListener("click", () => submit(true));
  confirmButton.addEventListener("click", () => submit(false));

  function setError(msg) {
    setText($("personError"), msg);
    show("personError", !!msg);
  }

  function setMode(next, job, candidate) {
    mode = next;
    const card = $("personCard");
    card.classList.toggle("attention", next === "choosing");
    setText($("personTitle"), m(next === "locked" ? "person.reconstructing" : "person.choose"));
    show("personConfirm", next === "choosing");
    show("personCount", next === "choosing");
    if (next !== "choosing") {
      for (const control of ["personJoinRow", "personJoinHelp", groupList, groupNext]) show(control, false);
    }
    if (next !== "locked") show("personReselect", false);
    show(card, next !== "hidden");
    if (next === "choosing") renderChoosing();
    else rows = [];
    if (next === "locked") renderLocked(job, candidate);
  }

  $("personReselect").addEventListener("click", () => bus.emit("person:reselect"));

  return {
    /** Called on every job poll. */
    update(job) {
      const cands = Array.isArray(job.candidates) ? job.candidates : [];
      const changedJob = jobId !== job.id;
      if (changedJob) {
        jobId = job.id;
        candidates = [];
        defaultIdx = previewIndex = lockedIndex = null;
        selected.clear();
        groups.length = 0;
        submitting = confirmed = false;
        suggested = null;

        joinTracks.checked = false;
        setError("");
      }
      if (job.status === "waiting_user" && cands.length) {
        if (mode !== "choosing" || changedJob) {
          candidates = cands;
          previewIndex = defaultIdx = job.default_person ?? 0;
          suggested = Array.isArray(job.suggested_join) && job.suggested_join.length >= 2
            ? job.suggested_join : null;
          if (suggested) {
            suggested.forEach(index => selected.add(index));
            previewIndex = suggested[0];
            joinTracks.checked = true;
          }
          setMode("choosing", job);
          bus.emit("person:candidates", { candidates, selected: [...selected], previewIndex });
        }
      } else if (job.selected_person != null) {
        const changed = mode !== "locked" || lockedIndex !== job.selected_person;
        lockedIndex = job.selected_person;
        if (changed) {
          const candidate = job.track_fragments?.length ? job.person_track
            : candidates[lockedIndex] ?? cands[lockedIndex] ?? job.person_track;
          setError("");
          setMode("locked", job, candidate);
          bus.emit("person:locked", { candidate, index: lockedIndex });
          candidates = [];
        }
        setText($("personTitle"), m(job.status === "done" ? "person.reconstructed"
          : job.status === "running" ? "person.reconstructing" : "person.selected"));
      } else if (mode !== "hidden") {
        setMode("hidden");
        bus.emit("person:candidates", { candidates: [], selected: [], previewIndex: null });
      }
      // Offer a re-pick once the job has finished (needs a re-run of detection).
      show("personReselect", mode === "locked" && ["done", "failed"].includes(job.status));
    },
    select: togglePerson,
    reset() {
      jobId = null; candidates = []; defaultIdx = previewIndex = lockedIndex = null;
      selected.clear();
      groups.length = 0;
      submitting = confirmed = false;
      suggested = null;

      joinTracks.checked = false;
      setError("");
      setMode("hidden");
      bus.emit("person:candidates", { candidates: [], selected: [], previewIndex: null });
    },
  };
}
