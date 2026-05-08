# Level One

H1 content — becomes an anchor (depth 1 ≤ max_heading_depth=4).

## Level Two

H2 content — becomes an anchor (depth 2).

### Level Three

H3 content — becomes an anchor (depth 3).

#### Level Four (boundary)

H4 is the deepest anchor with ``max_heading_depth=4``.

##### Level Five (sub-chunk)

This H5 heading is *below* the depth limit.  It should be treated as content
of the "Level Four" anchor, not its own anchor.

###### Level Six (sub-chunk)

H6 is also below the limit.  Treated as sub-chunk content of Level Four.
