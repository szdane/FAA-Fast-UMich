The paper that describes this work is called `paper_draft.pdf` and you should refer to it to understand the theory behind the proof approach. Before you do anything, ALWAYS read `paper_draft.pdf` and `fmcad22paper.pdf`.

The workflow here is described more in `pvs-examples/CLAUDE.md` but general engineering tip:

- The Python files take a spec and automatically generate PVS artifacts with Prooflite (scripting language for the proof tree)
- If you're working on generating code that makes PVS files, and don't know if a change to the Python code will work on the proof, modify the PVS file artifact first and then test to see if that change makes a difference.
- If you make changes, test against some older examples (some of examples 1 through 6, they get more complex as they go) to ensure there are no regressions.

When I talk about "cases" that is BOTH piecewise cases AND including "notches" i.e. places where the trajectory tangent line is parallel to a side of the polygon (read paper for more). So a parabola-then-linear function with only two piecewise subfunctions may have THREE cases if a rectangle moves along it, as the parabola may be split into two cases.
