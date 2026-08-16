/**
 * Jobs Modal Component (agent platform Phase 5)
 * Run on-demand jobs (skills with input/output contracts), view run history
 * and outputs. POST /api/jobs/run is also the future cron hook.
 */
import { api } from '../utils.js'

const jobsModal = () => ({
  open: false,
  jobs: [],
  runs: [],
  selectedJob: '',
  paramsText: '{}',
  running: false,
  lastRun: null,

  openModal() {
    this.open = true
    this.$store.ui.openJobs()
    this.load()
  },

  closeModal() {
    this.open = false
    this.$store.ui.closeJobs()
  },

  async load() {
    try {
      const [skillsData, runsData] = await Promise.all([
        api.get('/api/skills'),
        api.get('/api/jobs')
      ])
      this.jobs = (skillsData.skills || []).filter(s => !s.draft)
      this.runs = runsData.runs || []
      if (!this.selectedJob && this.jobs.length) this.selectedJob = this.jobs[0].name
    } catch (e) { console.error('[jobs] load error:', e) }
  },

  async runJob() {
    if (this.running || !this.selectedJob) return
    this.running = true
    this.lastRun = null
    let params = {}
    try {
      params = JSON.parse(this.paramsText || '{}')
    } catch (e) {
      this.$store.ui.showToast('Params must be valid JSON', 'error')
      this.running = false
      return
    }
    try {
      const data = await api.post('/api/jobs/run', { job: this.selectedJob, params })
      this.lastRun = data.run || null
      this.$store.ui.showToast(
        `Job '${this.selectedJob}' ${data.run?.status === 'completed' ? 'completed' : 'failed'}`,
        data.run?.status === 'completed' ? 'success' : 'warning'
      )
      await this.load()
    } catch (e) {
      this.$store.ui.showToast('Job failed: ' + e.message, 'error')
    } finally {
      this.running = false
    }
  },

  rerun(run) {
    this.selectedJob = run.job_name
    this.paramsText = JSON.stringify(run.params || {}, null, 2)
  },

  outputUrl(run) {
    if (!run?.output_path) return ''
    const parts = run.output_path.split('/')
    return '/outputs/jobs/' + parts[parts.length - 1]
  },

  statusBadge(status) {
    if (status === 'completed') return 'badge-success'
    if (status === 'failed') return 'badge-error'
    return 'badge-warning'
  }
})

export { jobsModal }
