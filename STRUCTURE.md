# Portfolio Directory Structure

This portfolio now uses a multi-page architecture with page-based navigation and a dedicated project gallery.

## Current layout

```
.
├── portfolio.html
├── Iljard_Xhani_CV.pdf
├── STRUCTURE.md
├── assets/
│   ├── brand/
│   ├── css/
│   │   └── main.css
│   ├── docs/
│   ├── icons/
│   ├── images/
│   ├── js/
│   │   └── main.js
│   └── videos/
├── docs/
│   ├── architecture/
│   └── workflows/
├── pages/
│   ├── README.md
│   ├── about.html
│   ├── contact.html
│   ├── process.html
│   ├── _template/
│   │   └── project-case-study.html
│   └── projects/
│       ├── index.html
│       ├── docchat-rag-production.html
│       ├── rag-evaluation-lab.html
│       ├── support-ai-triage.html
│       ├── sql-docs-hybrid-agent.html
│       ├── knowledge-base-builder.html
│       └── rag-for-teams-saas.html
├── presentations/
│   ├── _template/
│   ├── archive/
│   └── upcoming/
└── projects/
    ├── _template/
    ├── docchat-rag-production/
    ├── rag-evaluation-lab/
    ├── support-ai-triage/
    ├── sql-docs-hybrid-agent/
    ├── knowledge-base-builder/
    └── rag-for-teams-saas/
```

## Navigation model

Primary nav is page-based (not section-scroll):

- Home: `portfolio.html`
- Projects Gallery: `pages/projects/index.html`
- Process: `pages/process.html`
- About: `pages/about.html`
- Contact: `pages/contact.html`

## Shared assets

- Shared styling: `assets/css/main.css`
- Shared behavior: `assets/js/main.js`

## Project delivery contract

Each project folder uses:

- `projects/<project-slug>/case-study/`
- `projects/<project-slug>/assets/`
- `projects/<project-slug>/presentations/upcoming/`
- `projects/<project-slug>/presentations/archive/`

## Upcoming presentation naming

Use one folder per deck:

- `projects/<project-slug>/presentations/upcoming/YYYY-MM-DD_audience_topic/`

Example:

- `projects/docchat-rag-production/presentations/upcoming/2026-03-04_hiring-panel_system-overview/`

## Cleanup note

Legacy one-page anchor navigation and old placeholder project pages were replaced by the new multi-page structure.
