import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'dataset-desk',
  description:
    'Make and review visual datasets: find copies, near-copies and outliers, search by words or picture, and check that every description describes its image.',
  logoBadge: 'DD',
  brandName: 'dataset-desk',
  repoUrl: 'https://github.com/mcp-tool-shop-org/dataset-desk',
  footerText:
    'MIT Licensed — built by <a href="https://mcp-tool-shop.github.io/" style="color:var(--color-muted);text-decoration:underline">MCP Tool Shop</a>',

  hero: {
    badge: 'v0.2.0 · prototype',
    headline: 'A desk for your dataset,',
    headlineAccent: 'before it trains anything.',
    description:
      'Point it at a folder of training images. It finds the repeats, the odd ones out and the descriptions that only echo their prompt, and writes a report and a review queue. <strong>Nothing is removed automatically.</strong>',
    primaryCta: { href: '#usage', label: 'Get started' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      { label: 'Install', code: 'pip install dataset-desk' },
      { label: 'Index', code: 'dataset-desk index --project art/ --backend none --out desk-out/art' },
      { label: 'Report', code: 'dataset-desk report --project art/ --backend none --out desk-out/art' },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'What it checks',
      subtitle: 'Two defects spoil a training set quietly: repeated pictures, and descriptions that are not descriptions.',
      features: [
        {
          title: 'Duplicates, three ways',
          desc: 'Byte-identical copies by sha256, near-copies by a 256-bit dHash, and semantic repeats by embedding cosine. Each pair is listed for a person to judge.',
        },
        {
          title: 'A description gate with no model',
          desc: "Flags text that echoes the generation prompt, wraps the subject line, is too short, repeats another description, or runs past SigLIP2's 64 tokens.",
        },
        {
          title: 'A blind worksheet',
          desc: 'Each flagged image sits beside its nearest siblings, with no prompt and no old text in view, so the rewrite describes the picture.',
        },
        {
          title: 'Content and look, apart',
          desc: 'Outliers are ranked twice: by what the image shows, and by a 30-number look channel of palette, tone and native texture.',
        },
        {
          title: 'A lite tier with no torch',
          desc: 'numpy and Pillow only. Duplicates, the look channel, the gate and the report all run on a laptop CPU.',
        },
        {
          title: 'Polite on a shared machine',
          desc: 'CPU by default with capped thread pools; a GPU needs a grant id; it never writes into the project.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Usage',
      subtitle: 'Five commands. Every one reads the project and writes only under --out.',
      cards: [
        {
          title: 'Lite pass (no model)',
          code: 'dataset-desk index  --project art/ --backend none --out desk-out/art\ndataset-desk report --project art/ --backend none --out desk-out/art',
        },
        {
          title: 'With an embedding index',
          code: 'pip install "dataset-desk[models]"\ndataset-desk index  --project art/ --backend siglip2 --out desk-out/art\ndataset-desk search --project art/ --backend siglip2 --out desk-out/art \\\n  --text "a squat bog creature"',
        },
        {
          title: 'Duplicates as JSON',
          code: 'dataset-desk dupes --project art/ --backend siglip2 --out desk-out/art > dupes.json',
        },
        {
          title: 'Coverage of a subject list',
          code: 'dataset-desk coverage --project art/ --backend siglip2 --out desk-out/art \\\n  --subjects wanted.txt',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'tiers',
      title: 'Tiers',
      subtitle: 'Pick the backend your machine can carry.',
      columns: ['Backend', 'Needs', 'Gives you'],
      rows: [
        ['none', 'numpy, Pillow', 'copies, near-copies, look outliers, the description gate, the report'],
        ['siglip2', 'the models extra; SigLIP2 so400m', 'all of the above, plus search, coverage, semantic repeats, content outliers'],
        ['eg2', 'the models extra; EmbeddingGemma 2', 'the same, with joint text + picture search'],
      ],
    },
    {
      kind: 'data-table',
      id: 'trust',
      title: 'Trust',
      columns: ['Question', 'Answer'],
      rows: [
        ['What does it read?', 'The folder named by --project: images, records/*.json, wave metadata, PNG generation data.'],
        ['What does it write?', 'Only under --out. An --out inside the project is refused.'],
        ['Does it delete anything?', 'No.'],
        ['Network?', 'None in the lite tier. A model backend fetches its pinned weights on first use unless HF_HUB_OFFLINE=1.'],
        ['Telemetry?', 'None.'],
      ],
    },
  ],
};
