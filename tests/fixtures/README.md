# Drawer background regression

`drawer_handle_background.npz` contains the observed world-frame RGB-D cloud from
the retained panel-v2 failure, converted to float32 and masked outside the handle
region and a stride-4 cabinet sample. Other pixels are NaN. It preserves image
coordinates for the recorded normalized boxes; it contains no controller truth.

Source image hash and portable scene reconstruction are documented in
`evidence/drawer-handle-fix-20261025/README.md`. The dominant background height
must not hide the less populous, geometrically valid handle height.

`drawer_three_handles.npz` is the float32 agent-view RGB-D cloud from the closed
three-drawer diagnostic scene in `.local/combination-fix-20261026/drawer-cloud/`.
It tests ranking the observed bars, preserving an already-correct handle, and
finding a stationary vertical reference below the top drawer. No object or joint
truth is used by the controller. The source frame is archived in
`evidence/combination-fix-20261026/diagnostics/`.

`drawer_open_middle.npz` contains the fresh depth cloud after the middle drawer
was opened and a JSON record of the preceding visually/depth-verified opening.
Sources: `.local/combination-fix-20261026/drawer-close-depth/drawer-cloud.npy` and
`drawer-v2/drawer-chain/command-01/step-01/result.json`. Prior geometry only gates
the search; tests require finding a horizontal bar again in current depth and
reject a wrong level or empty observation.
