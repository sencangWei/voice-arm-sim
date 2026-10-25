# Drawer background regression

`drawer_handle_background.npz` contains the observed world-frame RGB-D cloud from
the retained panel-v2 failure, converted to float32 and masked outside the handle
region and a stride-4 cabinet sample. Other pixels are NaN. It preserves image
coordinates for the recorded normalized boxes; it contains no controller truth.

Source image hash and portable scene reconstruction are documented in
`evidence/drawer-handle-fix-20261025/README.md`. The dominant background height
must not hide the less populous, geometrically valid handle height.
