# Introduction

Standard ASCII heading — slugifies normally.

## Café

Heading with a non-ASCII accented character.  ``café`` slugifies to ``cafe``
(ASCII fold) or ``caf`` depending on the slugifier; neither is empty so no
hash fallback should be needed.

## 🎉

Emoji-only heading — after Unicode normalisation the slug is empty, so the
fallback ``section-<short-content-hash>`` applies.
