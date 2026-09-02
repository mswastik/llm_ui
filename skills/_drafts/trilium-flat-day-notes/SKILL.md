---
name: trilium-flat-day-notes
description: Configure Trilium Notes to create daily notes as direct children of Journal root (flat structure) instead of Year/Month hierarchy
---

<!-- reflection reason: This is a repeatable, multi-step Trilium configuration procedure that multiple users would benefit from. The user already has a working implementation (note hrDXqSk12sjv) that can be packaged as a reusable skill for Trilium 0.105+. -->

1. Create a Day Template note with desired daily note content
2. Create a backend script note (type: code, mime: application/javascript;env=backend) that:
   - Triggers via ~runOnNoteCreation relation
   - Detects day notes by #dateNote label
   - Uses api.transactional() with raw SQL on branches table to move note under Journal root
3. Configure Journal root note with these inheritable attributes:
   - label: calendarRoot = (empty)
   - label: datePattern = {isoDate}
   - label: label:dateNote = promoted,single,date
   - relation: dateTemplate → Day Template note
   - relation: runOnNoteCreation → Backend script note
4. Test by creating a new day note (Ctrl+Shift+D or calendar click) — it should appear directly under Journal root
