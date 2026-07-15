# Next.js Design System & UI Tokens

This system details the styling parameters used across the React components inside `web/` to match institutional academic aesthetics.

## Design Palettes

| Token | Hex Value | Role / Usage |
|---|---|---|
| `--primary` | `#1E3A5F` | Institutional Navy — Primary Scholarly Branding |
| `--secondary` | `#2563EB` | Active Blue — Buttons, Active Focus Borders |
| `--accent` | `#A16207` | Research Gold — Accents, Key Highlights, Gauges |
| `--background`| `#F8FAFC` | Scholarly light page background |
| `--foreground`| `#0F172A` | Primary text copy color |
| `--muted` | `#E9EEF5` | Muted background fills for tables & header blocks |
| `--border` | `#CBD5E1` | Slate borders for dividers and data cards |
| `--destructive`| `#DC2626` | Deep red for critical grounding hallucination alerts |

## Scholarly Typography

- **Headings & Display**: `Crimson Pro` (serif, scholarly weight for focus titles)
- **UI Elements & Data Copy**: `Atkinson Hyperlegible` (sans-serif, optimized for readability of dense paragraphs)
- **Imports**: Google Fonts link for both families:
  `https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@400;500;600;700&display=swap`

## Signature Layout Elements
- **Data-Dense Cards**: Rounded slate borders (`1px border-[#CBD5E1] rounded`), subtle box shadows, padding bounded between `8px–16px` for dense scanning.
- **Accessible Radar**: 6-axis Radar Chart representation, paired with a mandatory data table layout for full readability access.
