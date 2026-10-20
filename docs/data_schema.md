# Data schema (verified)

Generated 2026-10-20 23:32 UTC by `python -m src.data.verify_assets`.

## SkillSpan (`data/raw/github/SkillSpan/data/json/*.json`)

One JSON object per line: `idx` (int), `tokens` (list[str]), `tags_skill` (B/I/O per token), `tags_knowledge` (B/I/O per token), `source` ('tech' | 'house').
Counts: {'train': 4800, 'dev': 3174, 'test': 3569}.

Hugging Face copy present (`data/raw/hf/jjzha__skillspan`): ['.gitattributes', 'dev.json', 'README.md', 'train.json', 'test.json']

## Skill-Extraction-benchmark (`data/raw/github/Skill-Extraction-benchmark/data/*.csv`)

Columns: `sentence`, `span`, `sub_span`, `label` (ESCO 1.1.0 preferred label | 'LABEL NOT PRESENT' | 'UNDERSPECIFIED'); `techwolf_test_annotations.csv` has only `sentence`, `label`.
Rows per file: {('house', 'test'): 891, ('house', 'validation'): 231, ('tech', 'test'): 1164, ('tech', 'validation'): 293, ('techwolf', 'test'): 588}

## ESCO v1.1.0 (`data/raw/esco/1.1.0/en`)

`skills_en.csv` columns: ['conceptType', 'conceptUri', 'skillType', 'reuseLevel', 'preferredLabel', 'altLabels', 'hiddenLabels', 'status', 'modifiedDate', 'scopeNote', 'definition', 'inScheme', 'description']; 13,891 skills; files: ['ISCOGroups_en.csv', 'broaderRelationsOccPillar_en.csv', 'broaderRelationsSkillPillar_en.csv', 'conceptSchemes_en.csv', 'digCompSkillsCollection_en.csv', 'greenSkillsCollection_en.csv', 'ictSkillsCollection_en.csv', 'languageSkillsCollection_en.csv', 'occupationSkillRelations_en.csv', 'occupations_en.csv', 'researchSkillsCollection_en.csv', 'skillGroups_en.csv', 'skillSkillRelations_en.csv', 'skillsHierarchy_en.csv', 'skills_en.csv', 'transversalSkillsCollection_en.csv']

## Synthetic ESCO skill sentences (`data/raw/hf/TechWolf__Synthetic-ESCO-skill-sentences`)

Columns: ['sentence', 'skill'] → sentence column = `sentence`, skill column = `skill`; 138,260 rows; 13,826 distinct skills.
Example: {'sentence': 'the ideal candidate for this position should be able to advise customers on sewing patterns based on their needs', 'skill': 'advise customers on sewing patterns'}

## djinni_jobs (`data/raw/hf/lang-uk__recruitment-dataset-job-descriptions-english`)

Source columns: ['Position', 'Long Description', 'Company Name', 'Exp Years', 'Primary Keyword', 'English Level', 'Published', 'Long Description_lang', 'id', '__index_level_0__']
Mapping to our schema: {'title': 'Position', 'description': 'Long Description', 'company': 'Company Name', 'category': 'Primary Keyword', 'exp_years': 'Exp Years', 'english_level': 'English Level', 'published': 'Published'}
Rows: 141,897

## djinni_cvs (`data/raw/hf/lang-uk__recruitment-dataset-candidate-profiles-english`)

Source columns: ['Position', 'Moreinfo', 'Looking For', 'Highlights', 'Primary Keyword', 'English Level', 'Experience Years', 'CV', 'CV_lang', 'id', '__index_level_0__']
Mapping to our schema: {'title': 'Position', 'category': 'Primary Keyword', 'exp_years': 'Experience Years', 'english_level': 'English Level', 'moreinfo': 'Moreinfo', 'looking_for': 'Looking For', 'highlights': 'Highlights', 'cv': 'CV'}
Rows: 210,250

## open-apply-jobs (`data/raw/live_jobs`, snapshot 2026-10-20)

Columns: ['id', 'source_slug', 'title', 'apply_url', 'description_html', 'employment_type', 'department', 'locations', 'remote', 'posted_at', 'updated_at', 'salary_min', 'salary_max', 'salary_currency', 'salary_period', 'date', 'source']
Mapping: {'title': 'title', 'description_html': 'description_html', 'company': None, 'locations': 'locations', 'remote': 'remote', 'department': 'department', 'employment_type': 'employment_type', 'posted_at': 'posted_at', 'apply_url': 'apply_url', 'source': 'source'}
Snapshot rows: 1,162,694; data/AI/ML filtered rows: 21,157
