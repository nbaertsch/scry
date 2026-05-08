# Oversized File Marker

This small file is a template for the oversized-file test fixture.
The test dynamically creates a 6 MB version of this file at runtime.

Default ``index.max_file_size_bytes``: 5242880 (5 MB).
Files exceeding this limit are skipped with a warning; no anchors are created.
