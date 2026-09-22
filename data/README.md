# data/

The study's only input is the synthetic HazMat fixture, 33 rows in four CSV tables plus
the reference answers, and it ships inside the reference implementation at
`code/pathproof/hazpath/data/` (copied unmodified from the HazThread demo build; its
seeded defects are documented there). It is hashed file by file in `results/manifest.json`.
Nothing else is read. There is no customer data anywhere in this pack and none may be added.
