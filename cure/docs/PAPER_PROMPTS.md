# CURE Paper - Experimental Setup

## Object Removal Prompts (Imagenette Classes)

From the paper (Page 8, Table 4):
The paper tests removal of objects from Imagenette dataset subset:
- Cassette Player
- Chain Saw
- Church
- English Springer (dog)
- French Horn
- Garbage Truck
- Gas Pump
- Golf Ball
- Parachute
- Tench (fish)

**Prompt format:** Just the object name, sometimes with "a" or "the"

## Artist Removal Prompts (Figure 4, Table 1)

Classical Artists (5):
- Van Gogh
- Pablo Picasso
- Rembrandt
- Andy Warhol
- Caravaggio

Modern Artists (5):
- Kelly McKernan
- Thomas Kinkade
- Tyler Edlin
- Kilian Eng
- Ajin: DemiHuman

**Prompt format:** 20 prompts per artist
Examples: "{artist}", "a painting in the style of {artist}", "{artist} style", etc.

## NSFW/Nudity Removal

Concepts removed:
- Nudity
- Naked
- Nude
- Explicit

**Prompt format:** Direct concept names

## Key Findings from Paper

1. **Alpha values used:**
   - α = 2 for objects/artists
   - α = 5 for NSFW content

2. **Number of prompts:**
   - 20 prompts per concept (from prior work [34, 35])
   - Multiple variations per concept

3. **Retain concepts:**
   - Used for discriminative removal (Pdis = Pf - Pf @ Pr)
   - Helps preserve related concepts

4. **Results (Table 4 - Object Removal):**
   - Cassette Player: 0% accuracy (complete removal)
   - Chain Saw: 0%
   - Church: 4.2%
   - English Springer: 0%
   - French Horn: 0%
   - etc.

   For unrelated classes: ~79% accuracy (preserved)

5. **Efficiency (Table 5):**
   - Modification time: ~2 seconds
   - Inference time: 7.06 s/sample (same as original SD)
   - Model modification: 2.23%
