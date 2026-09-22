# interpretDB vs. a raw LLM — a two-hour test

**For:** the developer running it.
**You do not need any Python from us.** Everything here is SQL, a prompt to
paste, and a table to fill in. Use the real interpretDB, not a simulation.

Three questions. Each one has a **plausible wrong answer** that a model will
give you confidently, with no error and no warning. The whole point of the test
is to find out which of our layers stops it.

---

## Step 1 — load the data (5 minutes)

Four CSVs, 33 rows total, all synthetic. They are in
`pathproof/hazpath/data/`, or in the original HazThread demo build.

```sql
CREATE TABLE materials_hmid (
  hmid_id text, nsn text, material_name text, cas text,
  manufacturer text, mfr_state text, shelf_life_months int,
  uom text, hazard_class text);

CREATE TABLE shops (
  shop_id text, shop_name text, bldg text,
  substitute_authorized_per_TO text, state text);

CREATE TABLE supply_transactions (
  supply_id text, fy text, date date, nsn text,
  action text, qty numeric, unit_cost numeric, shop_id text);

CREATE TABLE disposal_dtid (
  dtid text, fy text, date date, nsn text, qty numeric,
  reason text, writeoff_usd numeric, disposal_cost_usd numeric);
```

```bash
psql -d hazmat -c "\copy materials_hmid      FROM 'materials_hmid.csv'      CSV HEADER"
psql -d hazmat -c "\copy shops               FROM 'shops.csv'               CSV HEADER"
psql -d hazmat -c "\copy supply_transactions FROM 'supply_transactions.csv' CSV HEADER"
psql -d hazmat -c "\copy disposal_dtid       FROM 'disposal_dtid.csv'       CSV HEADER"
```

**Do not clean the data.** It is dirty on purpose.

---

## Step 2 — RUN A: the raw model, no context

Paste this into the LLM. Nothing else. No Context Core, no Lexical Core, no
Persona Graph, no knowledge graph, no examples.

```
You are given a PostgreSQL schema. Write one SQL query that answers the
question. Return only SQL.

TABLES
  materials_hmid(hmid_id, nsn, material_name, cas, manufacturer,
                 mfr_state, shelf_life_months, uom, hazard_class)
  shops(shop_id, shop_name, bldg, substitute_authorized_per_TO, state)
  supply_transactions(supply_id, fy, date, nsn, action, qty, unit_cost, shop_id)
  disposal_dtid(dtid, fy, date, nsn, qty, reason, writeoff_usd, disposal_cost_usd)

QUESTION
  <paste one question from Step 3>
```

Run the SQL it gives you. **Write down the number it returns.** Do not fix the
SQL. Do not tell the model it is wrong. Do not retry. A wrong answer here is the
result we are looking for.

---

## Step 3 — the three questions

**Q1.** *What did we spend ordering the chromate primer (NSN 8010-01-555-1234)
across FY24 to FY26?*

**Q2.** *How many distinct states do the shops sit in?*

**Q3.** *How much primer did the Fuel Cell shop order in FY26?*

---

## Step 4 — RUN B, C, D through interpretDB

Same three questions, four times each. Record every one.

| Run | Configuration | What to switch off |
|---|---|---|
| **A** | Raw LLM | everything — Step 2 above |
| **B** | interpretDB, **context off** | Context Core, Lexical Core, Persona Graph disabled. Model still sees only the schema |
| **C** | interpretDB, **context on**, memory empty | full grounding, but wipe the verified-path/query cache first |
| **D** | interpretDB, **context on**, memory seeded | run Q1 and Q2 first so they are in memory, then ask Q3 |

For each run record: **the SQL, the number returned, and how long it took.**

---

## Step 5 — fill this in and send it back

| | Q1 ordered value | Q2 distinct states | Q3 Fuel Cell |
|---|---|---|---|
| **A** raw LLM | | | |
| **B** no context | | | |
| **C** context, no memory | | | |
| **D** context + memory | | | |
| **Correct** | **537,600.00** | **1** | **refuse to answer** |

Also note, for each cell: did anything warn you, or did it just return a number?

---

## The answer key, and why each trap works

Read this **after** you have filled the table, not before.

### Q1 — the duplicate trap. Wrong answer: **672,000.00**

`supply_transactions` documents `supply_id` as unique. It is not. Three ids
repeat: `SUP-00004`, `SUP-00008`, `SUP-00012`.

```sql
-- what a model writes
SELECT SUM(qty * unit_cost) FROM supply_transactions
WHERE fy IN ('FY24','FY25','FY26');              -- 672000.00   WRONG

-- what is correct
SELECT SUM(qty * unit_cost) FROM (
  SELECT DISTINCT supply_id, qty, unit_cost FROM supply_transactions
  WHERE fy IN ('FY24','FY25','FY26')) t;         -- 537600.00   right
```

The wrong answer is 25% high. Nothing errors. Both queries are valid SQL and
both look reasonable in a review.

### Q2 — the spelling trap. Wrong answer: **3**

`state` is spelled `California`, `CA` and `Calif.` in six rows.

```sql
SELECT COUNT(DISTINCT state) FROM shops;         -- 3   WRONG
```

Every shop is in California. There is one state. Any per-state grouping built on
this is wrong in the same way, and the error grows as you group by more things.

### Q3 — the coverage trap. Wrong answer: **0**

Shop `S06` (Fuel Cell) is in the roster and appears in **zero** transactions.

```sql
SELECT COALESCE(SUM(qty * unit_cost), 0) FROM supply_transactions
WHERE shop_id = 'S06' AND fy = 'FY26';           -- 0   WRONG
```

Zero is not the answer. Zero means "they ordered none." The truth is "we have no
data for this shop," and those are different sentences to a maintainer deciding
whether to reorder. **The correct behaviour is to refuse and say why.**

This is the most important of the three. The first two are arithmetic errors.
This one is a system claiming knowledge it does not have.

---

## What we expect to see, and what would surprise us

| Prediction | If it comes out otherwise |
|---|---|
| A gets Q1 and Q2 wrong | A model that dedups unprompted is worth knowing about. Save the prompt |
| B ≈ A | If B fixes anything, grounding is doing work we have not accounted for |
| C gets Q1 and Q2 right | If C still misses, the Context Core is not carrying the uniqueness or the synonym set, and that is a bug to file |
| C or D refuses Q3 | **If it returns 0, that is the most important finding in the test.** Tell us immediately |
| D is faster or cheaper than C on Q3 | If D is slower, the retrieval is not paying for itself yet |

**Send back the filled table and the SQL from every cell.** Wrong answers are as
useful as right ones here. Do not tidy anything up before sending it.

---

## One thing to be careful about

Run **C** with the cache genuinely empty. If a previous run left an entry, C and
D become the same test and the comparison is worthless. Clear it, then confirm
it is clear, then start.

---

## Optional: the Python

`pathproof.tar.gz` contains a reference implementation of the mechanism — the
primitive vocabulary, the path memory and the transfer logic — with 19 tests. It
is a design document you can execute, and it is useful if you want to see the
shape of what interpretDB should be doing internally.

**It is not needed for this test.** This test measures the real product.
