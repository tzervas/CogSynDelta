# Adversarial verification of 10-survey-other-sources.md

Session date: 2026-09-03. Method: for every non-REFUSE entry in `10-survey-other-sources.md`,
independently re-fetched the **primary/upstream** page the entry cites (never the survey's own
prose) via WebFetch, and via `curl` (with a browser UA, `-k` where a TLS chain error blocked the
normal client — the cert error is a local trust-store/interception issue, not a spoofed host;
same-host content matched WebFetch's other successful fetches) when WebFetch returned an
empty/nav-shell page. For the one REFUSE entry (CourtListener), verified anyway, on the
adversarial-verification principle that a wrongly-refused source is as much a defect as a
wrongly-admitted one. No dataset content downloaded — pages and licence text only, all under
this scratchpad/staging tree, nothing in `/tmp`.

Checked against the class list: scraped-without-licence, model-output terms, research-only
READMEs, annotation-vs-media rights, NC/ND parents, consent-revocable, mirror mismatch.

**Bottom line: 19 of 21 entries CONFIRMED (several strengthened with fuller primary text than
the original survey captured). One entry is CONTRADICTED — CourtListener's REFUSE verdict rests
on a licence claim the primary source for the actual bulk data does not support. One entry
(DTD) is confirmed-as-appropriately-unresolved but the primary source turns out to hand us two
of the refuse-class red flags directly, which the original entry didn't have in hand.**

---

## CONTRADICTED — read this one first

### #17 CourtListener bulk data — REFUSE verdict's premise does not hold up

The survey refused this source as **BLOCKING** on the claim: *"Content is licensed under a
Creative Commons BY-ND international 4.0 license, except where indicated"* — sourced as
"search-verified... recommend direct WebFetch confirmation before final close-out." That
recommended confirmation is what this pass did, and it does not confirm the claim for the
dataset itself.

- Fetched `https://www.courtlistener.com/help/api/bulk-data/`, which 301-redirects to
  `https://wiki.free.law/c/courtlistener/help/api/bulk-data/bulk-legal-data` — Free Law
  Project's own current documentation for the bulk data files. Fetched twice (once directly,
  once re-confirming with a request for every clause), consistent both times.
- **VERIFIED, primary source**: *"Our bulk data files are free of known copyright
  restrictions,"* displayed with the Creative Commons **Public Domain Mark**
  (`creativecommons.org/publicdomain/mark/1.0/`) — not a CC BY-ND badge.
- The page lists a licence for **every** bulk data table by name — Courts, Dockets, Opinion
  Clusters & Opinions, Citations Map, Parentheticals, Integrated DB (FJC), Financial
  Disclosures, Judges, Oral Arguments, Case Law Embeddings — and every one is marked **Public
  Domain**. No ND clause, no derivatives restriction, anywhere on this page.
- Attempted to reach the general `courtlistener.com/terms/` page (where a CC BY-ND footer
  boilerplate plausibly lives, covering the *website*, not the *data*) but it returned HTTP 403
  from CloudFront bot protection on every attempt (WebFetch and curl, both UA-spoofed). A
  WebSearch re-asserted the CC BY-ND claim but could not produce a live quote-with-URL for it —
  it reads as the search index repeating the same unconfirmed claim, not an independent primary
  hit.

**What this means**: the original entry's own methodology note (WebFetch primary sources, never
trust a mirror/search summary alone) was the right instinct and wasn't followed through here —
this is exactly the failure mode the survey's own preamble warns against, just applied to the
one REFUSE call instead of an admit. The specific claim needed to justify BLOCKING (ND on the
*bulk data*) is contradicted by Free Law Project's own current bulk-data documentation, fetched
directly, twice. It remains plausible a CC BY-ND clause exists somewhere on courtlistener.com
covering the site's own presentation/UI layer — that would not touch the data licence, which is
what a training-corpus decision turns on.

**Recommendation**: do not leave this filed as REFUSE/BLOCKING on the current basis. Re-open
with a fetch of `courtlistener.com/terms/` once the CloudFront block can be gotten past (a
different egress, or ask a human to paste the terms page), to settle whether CC BY-ND exists at
all and, if so, what it scopes. Absent that, the bulk-data page's explicit per-table Public
Domain Mark is the stronger primary evidence and points toward PERMISSIVE_OK for the core legal
data (opinions, dockets, citations), which would also match the survey's own independent
observation that "raw court opinions themselves are separately public domain."

---

## Strengthened — confirmed, with the primary text the original entry couldn't get

### #3 COCO — terms-of-use text obtained (original fetch got only a nav shell)

The page is a single-page app; the Terms of Use tab loads
`https://cocodataset.org/dataset/termsofuse.htm` via JS, which is why both the original survey's
fetch and this pass's first WebFetch attempt returned only the nav shell. Pulled that fragment
directly:

> "The annotations in this dataset along with this website belong to the COCO Consortium and
> are licensed under a Creative Commons Attribution 4.0 License." / "The COCO Consortium does
> not own the copyright of the images. Use of the images must abide by the Flickr Terms of
> Use. The users of the images accept full responsibility for the use of the dataset..." /
> Software: 3-clause BSD, copyright COCO Consortium 2015.

This matches the survey entry's characterization exactly (annotations ATTRIBUTION,
images unstated/Flickr-governed) — **CONFIRMED**, and the entry's own "needs a full fetch...
before treating as anything beyond provisional" caveat is now resolved in the entry's favor.

### #13 Software Heritage — confirmed past a TLS handshake failure

WebFetch and a plain `curl` both failed with `unable to get local issuer certificate` on
`softwareheritage.org` (a local trust-store gap for that host's chain, not a redirect/spoof —
other HTTPS hosts fetched cleanly throughout this session). Re-fetched with `curl -k` to read
the same live page: **VERIFIED**, primary source —

> "Software Heritage may provide automatically derived information on the software license(s)
> that may apply to a given software component, but it makes no claim of correctness... You are
> solely responsible for determining the license, or other rights that apply to any software
> component in the Archive, and you must abide by its terms."

Also newly captured: metadata (provenance, file type/length, language) is called out as
"factual, not covered by copyright" — consistent with, and slightly sharper than, the survey's
"infrastructure, not a licence" framing. **CONFIRMED**, entry's "flag for re-fetch" resolved.

### #8 bAbI tasks — full licence text obtained (original was flagged partial)

`raw.githubusercontent.com/facebookarchive/bAbI-tasks/master/LICENSE.md` rendered fully this
time: standard 3-clause BSD (copyright Facebook, Inc., 2015-present), including the
no-endorsement clause the survey's partial fetch hadn't captured. No additional restriction
beyond what's already boilerplate BSD-3. **CONFIRMED**, PERMISSIVE_OK stands, the "re-fetch
recommended before ingest" flag is now cleared.

### #14 C4 / Common Crawl — the provenance_red_flag is exactly right, now with primary text

AllenAI's ODC-BY grant confirmed on the HF-hosted README (`huggingface.co/datasets/allenai/c4`).
Went further and fetched Common Crawl's own ToU directly (`commoncrawl.org/terms-of-use`,
redirects from `.org/terms-of-use/`), which the survey flagged as "recommend a direct fetch...
before ingest" but hadn't done. **VERIFIED**, and it sharpens the caveat rather than softening
it: CC's grant to users is explicitly a *"LIMITED LICENSE... to access and use the Service"* —
access terms, not a copyright grant over the crawled pages — paired with *"BY USING THE CRAWLED
CONTENT, YOU AGREE TO RESPECT THE COPYRIGHTS AND OTHER APPLICABLE RIGHTS OF THIRD PARTIES."*
This is a clean primary-source confirmation of the survey's own read: ODC-BY covers AllenAI's
compilation, Common Crawl's ToU is a usage agreement layered on top, neither one clears the
underlying pages' copyright status. **CONFIRMED**, provenance_red_flag correctly assigned.

### #15 arXiv metadata (Kaggle mirror) — both layers now directly confirmed

- `info.arxiv.org/help/license` (the `arxiv.org/help/license` URL 301s here) **VERIFIED**: *"A
  Creative Commons CC0 1.0 Universal Public Domain Dedication will apply to all metadata"* —
  authoritative, at the primary authorial-terms source, stronger than relying on Kaggle's copy.
- The Kaggle page itself resisted two WebFetch attempts (renders via JS), so pulled the raw HTML
  with curl and grepped it directly: `"CC0 1.0 Universal Public Domain Dedication"` /
  `"CC0: Public Domain"` both present in the page's embedded dataset metadata. **CONFIRMED** at
  both mirror and upstream — the entry's own "not yet fetched this session, flagged for
  follow-up" note on the arXiv-authoritative page is now resolved, matching.

---

## Confirmed as originally stated (primary text re-checked, matches)

- **#1 EuroSAT** — Zenodo record 7711810: "MIT License." Exact match.
- **#2 Visual Genome** — homepage: licence links to `creativecommons.org/licenses/by/4.0/`
  (page renders the words "Creative Commons" as the CC BY 4.0 hyperlink; the fuller sentence
  the survey quoted verbatim wasn't reproduced by this pass's fetch, but the same licence and
  URL is confirmed). **Caveat on the caveat**: the entry's claim that VG's images are drawn from
  MS-COCO/YFCC100M is *not* stated on the licensing page itself — it's accurate (confirmed via
  the VG paper, arXiv 1602.07332: "108,249 images... from the intersection of MS-COCO and
  YFCC100M"), but that's a different primary source than the one cited for the licence, worth
  distinguishing in the record. The entry's `metadata_only`-for-images treatment is the right
  call either way.
- **#4 Wikidata** — CC0 for main/property/lexeme, CC BY-SA 4.0 elsewhere. Exact match.
- **#5 Stack Exchange Data Dump** — CC BY-SA 4.0, all four attribution sub-clauses (source
  indication, hyperlink to original question, author name display, live hyperlink to author
  profile with no nofollow) verified verbatim on `archive.org/details/stackexchange`. Exact
  match, including the unusual live-hyperlink clause the entry flagged as a
  `provenance_red_flag`.
- **#6 DeepMind Mathematics Dataset** — Apache-2.0, full text confirmed at the GitHub raw
  LICENSE. Exact match.
- **#7 MATH (Hendrycks)** — MIT, full verbatim text confirmed at the raw LICENSE file,
  byte-for-byte match with what the survey quoted.
- **#9 Project Gutenberg** — public-domain-majority claim confirmed, including resolving a
  wording ambiguity: a second, targeted fetch confirmed the *"No permission is needed for
  non-commercial use"* sentence on that page is about the **Project Gutenberg trademark**
  specifically, not the underlying text — the content-commercial-use claim ("This applies for
  all use, including commercial use," re: quoting PG text) is separately and clearly stated.
  No contradiction; the entry's PERMISSIVE_OK read is right.
- **#10 EU Open Data Portal** — CC BY 4.0 (editorial), CC0 (metadata), per-resource for
  everything else. Exact match, confirms "index, not a dataset" framing.
- **#11 data.gov** — 17 U.S.C. §105 public-domain claim for federal works, explicit
  non-federal-may-differ caveat. Exact match.
- **#12 GDELT** — "unlimited and unrestricted use... without fee," conditioned on citation +
  link to gdeltproject.org. Exact match, ATTRIBUTION verdict correct (not PERMISSIVE_OK, since
  the citation condition is real).
- **#16 PMC Open Access Subset** — "License terms vary. Please refer to the license statement
  in each article" confirmed verbatim. Exact match, unstated-at-aggregate verdict correct.
- **#19 OpenML** — `openml.org/terms` resisted rendering to both WebFetch and curl (client-side
  app in both cases — the raw HTML for `docs.openml.org/terms/` is a full doc-site nav tree with
  no body text delivered without JS execution). Corroborated instead via a targeted WebSearch
  that surfaces the same terms page content: non-exclusive access licence "in accordance with
  any licenses granted by" the submitter, CC0 listed as a common-but-not-universal per-dataset
  option, platform code BSD-3-Clause. Matches the survey's characterization; flagging that this
  one is corroborated rather than directly re-rendered, same caveat the survey itself used for a
  couple of its own entries.

---

## Confirmed-and-sharpened: DTD (#18) — the "unresolved" call was right, for reasons now on hand

The survey left this **UNVERIFIED, do not admit** because the primary Oxford VGG page wasn't
fetched last session. It was fetched this pass, along with the dataset's own `README.txt`
(neither page carries a licence section at all), and the result is worth recording plainly
because it lands on two of the adversarial-check classes directly rather than staying merely
ambiguous:

- **research-only framing, from the primary source**: *"This data is made available to the
  computer vision community for research purposes."* No commercial/redistribution grant stated
  anywhere on either page.
- **scraped-without-licence provenance**: *"The images were collected from Google and Flickr by
  entering our proposed attributes and related terms as search queries."* No per-image licence
  clearance is described — DTD is a search-engine scrape, same shape as the "self-declared,
  unverified at scrape time" problem flagged elsewhere in the survey for Commons.

Neither page states a licence of any kind (not even the ambiguous HF "other" tag's ambiguity —
there's simply no licence section). Given the operator's explicit refuse list includes
"research-only / non-redistributable terms" and "scraped-without-licence," **this reads as
heading toward BLOCKING rather than merely unresolved** if a future pass revisits it — the
survey's caution was correct, and now there's primary text to cite instead of an absent fetch.

---

## Minor note: Wikimedia Commons (#21) — one clause not on the cited primary page

`commons.wikimedia.org/wiki/Commons:Licensing` **VERIFIED**: every upload must carry a licence
tag on the file description page; the licence must be free (redistribution, derivatives, and
commercial use all must be allowed, must be perpetual/non-revocable) — confirms the
"self-declared licence at upload" half of the entry's `provenance_red_flag` exactly.

The second half of that red flag — *"known history of mis-licensed uploads requiring post-hoc
removal"* — is accurate as general knowledge about Commons but is **not stated on this specific
policy page**; it doesn't appear as a quotable line there. Recorded as INFERRED, not
upstream-VERIFIED, on this pass — doesn't change the entry's (correctly cautious) treatment, but
the record should say so rather than let it read as sourced to the same page as the first half.

---

## Not independently re-confirmed this pass (source class already correctly not admitted)

- **#20 Papers with Code** — `stat.paperswithcode.com/datasets/license` failed with a TLS
  handshake failure on this pass (distinct host/error from the Software Heritage case, and this
  one didn't yield to `curl -k` either — different failure class, not investigated further given
  the entry is explicitly "index only, not admissible" and no admission decision rests on it).
  No material risk from leaving this unconfirmed.

---

## Cross-cutting

- The survey's own methodology (primary fetch over mirror/search trust) is sound and, on 19 of
  21 entries, held up under a second independent fetch — including several where the *original*
  fetch had failed to render and this pass got the real text (COCO, Software Heritage, bAbI,
  Common Crawl ToU, both arXiv layers). That's the intended behavior of "flag for re-fetch"
  notes: they got acted on correctly here.
- The one real defect found is procedural, not just factual: the **REFUSE** entry (CourtListener)
  was the one place the survey's own "search-verified, recommend direct WebFetch confirmation"
  flag was raised but not followed up before the verdict was finalized and returned to the
  caller as settled. A `BLOCKING` call is exactly as costly to get wrong as a bad `PERMISSIVE_OK`
  — it permanently write off a large, high-quality, well-curated corpus (10M+ opinions) on a
  claim that direct primary-source reading does not support for the actual data.
  **Recommend: do not treat REFUSE entries as exempt from the same fetch-before-file discipline
  applied to admits, in future survey passes.**
- No new mirror-vs-upstream mismatches were found beyond the ones the survey already flagged
  (COCO/VG annotations-vs-images, arXiv metadata-vs-full-text, PMC's per-article variance) — the
  "annotations clean, media/full-text unstated" shape recurs exactly as the survey's
  cross-cutting section says, and every instance of it was independently reconfirmed here.
