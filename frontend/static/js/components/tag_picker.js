/**
 * Capability picker — one searchable multi-select shared by the agent form's
 * MCP servers / custom tools / skills allow-lists.
 *
 * The backend treats an EMPTY list as "allow everything", which is the right
 * semantic and stays untouched; this component only replaces the old checkbox
 * grids that could not scale or filter. So the picker always states, in words,
 * whether the list is unrestricted or an explicit subset.
 *
 * Usage (bound to an existing reactive array on the parent scope):
 *
 *   <div x-data="capabilityPicker({
 *          id: 'mcp',
 *          options: () => pickerMcpOptions,
 *          value: () => formData.enabled_mcp_servers,
 *          assign: (v) => { formData.enabled_mcp_servers = v },
 *          searchPlaceholder: 'Search servers…',
 *          unrestrictedLabel: 'All connected servers (default)',
 *        })">
 *
 * `options` / `value` / `assign` are thunks resolved at call time, so the
 * picker keeps working when the parent replaces formData wholesale (open the
 * edit form for another agent) instead of holding a stale array reference.
 */
function capabilityPicker(config) {
  const opts = config || {}
  return {
    open: false,
    query: '',
    tab: null,

    // ─── Wiring to the parent scope ────────────────────────
    _options() { return (opts.options ? opts.options() : []) || [] },
    _value() { return (opts.value ? opts.value() : []) || [] },
    _assign(list) { if (opts.assign) opts.assign(list) },

    // ─── Derived state ─────────────────────────────────────
    get selected() { return this._value() },
    get unrestricted() { return this.selected.length === 0 },
    get allGroups() {
      const seen = []
      for (const o of this._options()) if (!seen.includes(o.group || '')) seen.push(o.group || '')
      return seen
    },
    get activeGroups() {
      const groups = this.allGroups
      return groups.length > 1 ? groups : []
    },
    get visibleOptions() {
      const q = this.query.trim().toLowerCase()
      const sel = new Set(this.selected)
      return this._options().filter(o => {
        if (q) {
          const hay = `${o.label || ''} ${o.value} ${o.description || ''}`.toLowerCase()
          if (!hay.includes(q)) return false
        }
        if (this.tab && (o.group || '') !== this.tab) return false
        return true
      })
    },
    // Selected items that no longer exist (server removed / skill uninstalled)
    // stay visible so saving cannot silently drop them.
    get orphaned() {
      const known = new Set(this._options().map(o => o.value))
      return this.selected.filter(v => !known.has(v))
    },
    get summaryText() {
      if (this.unrestricted) return opts.unrestrictedLabel || 'All (default)'
      return `${this.selected.length} selected`
    },

    // ─── Actions ───────────────────────────────────────────
    isSelected(value) { return this.selected.includes(value) },
    toggle(value) {
      const next = this.isSelected(value)
        ? this.selected.filter(v => v !== value)
        : [...this.selected, value].sort()
      this._assign(next)
    },
    remove(value) { this._assign(this.selected.filter(v => v !== value)) },
    clearAll() { this._assign([]) },
    selectVisible() {
      const merged = new Set(this.selected)
      for (const o of this.visibleOptions) merged.add(o.value)
      this._assign([...merged].sort())
    },
    toggleGroup(group) {
      const inGroup = this._options().filter(o => (o.group || '') === group).map(o => o.value)
      const allOn = inGroup.every(v => this.selected.includes(v))
      this._assign(allOn
        ? this.selected.filter(v => !inGroup.includes(v))
        : [...new Set([...this.selected, ...inGroup])].sort())
    },
    isGroupComplete(group) {
      const inGroup = this._options().filter(o => (o.group || '') === group)
      return inGroup.length > 0 && inGroup.every(o => this.selected.includes(o.value))
    },
    focusSearch() {
      this.$nextTick(() => {
        const el = this.$refs.search
        if (el) el.focus()
      })
    },
    togglePanel() {
      this.open = !this.open
      this.query = ''
      this.tab = null
      if (this.open) this.focusSearch()
    },
    close() { this.open = false },
    labelFor(value) {
      const o = this._options().find(x => x.value === value)
      return o ? (o.label || o.value) : value
    },
  }
}

export { capabilityPicker }
