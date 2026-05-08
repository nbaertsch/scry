# Main Title

Opening paragraph under the H1 anchor.

## Section One

Content of section one.  This should become its own anchor.

### Subsection A

Content of subsection A.  H3 is within max_heading_depth=4.

#### Deep Section

Content of the fourth-level section — the boundary for default max_heading_depth.

##### Level Five

This heading is *below* `max_heading_depth=4` and should be treated as content
of the nearest ancestor anchor (Deep Section), NOT its own anchor.

###### Level Six

Also below the depth limit; sub-chunk of Deep Section.

## Section Two

Second H2 section with its own content.

### Subsection B

Another H3 under Section Two.
