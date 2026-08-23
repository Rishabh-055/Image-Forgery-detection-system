---
name: Forensic Precision
colors:
  surface: '#101319'
  surface-dim: '#101319'
  surface-bright: '#363940'
  surface-container-lowest: '#0b0e14'
  surface-container-low: '#191c22'
  surface-container: '#1d2026'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2eb'
  on-surface-variant: '#b9cac9'
  inverse-surface: '#e1e2eb'
  inverse-on-surface: '#2e3037'
  outline: '#839493'
  outline-variant: '#3a4a49'
  surface-tint: '#00dddd'
  primary: '#ffffff'
  on-primary: '#003737'
  primary-container: '#00fbfb'
  on-primary-container: '#007070'
  inverse-primary: '#006a6a'
  secondary: '#ffb3ae'
  on-secondary: '#68000b'
  secondary-container: '#ad031a'
  on-secondary-container: '#ffb8b2'
  tertiary: '#ffffff'
  on-tertiary: '#003739'
  tertiary-container: '#79f5fb'
  on-tertiary-container: '#007074'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#00fbfb'
  primary-fixed-dim: '#00dddd'
  on-primary-fixed: '#002020'
  on-primary-fixed-variant: '#004f4f'
  secondary-fixed: '#ffdad7'
  secondary-fixed-dim: '#ffb3ae'
  on-secondary-fixed: '#410004'
  on-secondary-fixed-variant: '#930014'
  tertiary-fixed: '#79f5fb'
  tertiary-fixed-dim: '#59d8de'
  on-tertiary-fixed: '#002021'
  on-tertiary-fixed-variant: '#004f52'
  background: '#101319'
  on-background: '#e1e2eb'
  surface-variant: '#32353c'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.5'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.1em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: '1.2'
spacing:
  unit: 4px
  gutter: 16px
  margin-sm: 16px
  margin-md: 32px
  margin-lg: 48px
  comparison-gap: 2px
---

## Brand & Style

The design system is built for a high-stakes "Digital Forensics" and "Modern Laboratory" environment. The UI must evoke a sense of clinical accuracy, technical authority, and immediate clarity. The brand personality is analytical, cold, and unwavering—designed for investigators and data scientists who require a focused, distraction-free workspace.

The style is a hybrid of **Modern Minimalism** and **Technical Brutalism**. It utilizes high-contrast accents against deep backgrounds to guide the eye toward anomalies in image data. Visual noise is eliminated to prioritize the integrity of the evidence being analyzed.

## Colors

The palette is strictly functional. The base is a deep charcoal (#0E1117) to reduce eye strain during long investigation sessions and provide a neutral backdrop for color-sensitive heatmaps.

- **Forensic Cyan (#00FFFF)**: Used for active states, successful detections, and scanning progress indicators. It represents "Light" or "Discovery."
- **Danger Red (#FF4B4B)**: Reserved exclusively for high-probability splicing detections, system errors, and critical score indicators.
- **Surface & Borders**: Surfaces use a slightly lighter charcoal (#1C2028) to create subtle separation, while borders are sharp and low-opacity to maintain a technical grid feel.

## Typography

This design system utilizes a dual-font strategy to balance readability with technical flavor. 

**Inter** is the primary interface font, providing a neutral and modern foundation for menus, settings, and general UI copy. **JetBrains Mono** is utilized for all data-heavy outputs, metadata, file paths, and confidence scores. 

All labels should be rendered in `label-caps` (JetBrains Mono) to mimic laboratory labeling equipment. Large headlines should use tight letter spacing to appear more compact and "engineered."

## Layout & Spacing

The layout is governed by a **fixed 12-column grid** on desktop to ensure precise alignment of side-by-side analysis tools. 

- **Side-by-Side Comparisons**: Images are separated by a minimal 2px "comparison-gap" to allow the eye to track splicing lines across the boundary without interruption.
- **Workbenches**: The central UI area should be flanked by collapsible sidebars (320px) for metadata and toolsets.
- **Rhythm**: A 4px baseline grid ensures that all technical data aligns vertically across different columns, maintaining the "spreadsheet-like" precision required for forensic work.

## Elevation & Depth

This design system avoids traditional shadows in favor of **Tonal Layers** and **Low-Contrast Outlines**. 

Depth is communicated through brightness:
- **Level 0 (Background)**: #0E1117 (The base canvas).
- **Level 1 (Panels)**: #1C2028 (Sidebar and header containers).
- **Level 2 (Active/Hover)**: #2D333D (Interactive elements).

Borders are the primary method of separation. Use 1px solid lines in `border_color_hex` for all containers. For high-priority data points, a 1px `primary_color_hex` border may be used to "illuminate" the panel.

## Shapes

The shape language is strictly **Sharp (0px)**. 

In a digital forensics environment, rounded corners are perceived as too "soft" or "consumer-grade." Sharp corners reinforce the grid system, echo the rectangular nature of pixels, and maximize screen real estate for technical data density. Every button, input, and image container must have absolute 90-degree angles.

## Components

### Comparison Gauges
Score indicators are visualized as horizontal segmented bars rather than circular dials. Segments should light up in **Forensic Cyan** for low-risk and **Danger Red** for high-risk splicing probability.

### Technical Buttons
Buttons are text-based with JetBrains Mono. Primary buttons have a solid Cyan background with black text. Secondary buttons are ghost-style with a Cyan 1px border.

### Splicing Heatmaps
Heatmaps are overlaid on source images with a 40% opacity. They should use a "Jet" or "Turbo" color scale where high-intensity (spliced) areas glow in vivid Red/Yellow against the Cyan/Blue of the original image data.

### Metadata Lists
Lists of EXIF data should use alternating row highlights (Zebra striping) for readability. Keys are shown in a muted gray, while values are shown in pure white JetBrains Mono.

### Inputs
Input fields are "Underline" style or "Full Border" style with 0px radius. The cursor and focus state must always be the **Forensic Cyan** (#00FFFF).