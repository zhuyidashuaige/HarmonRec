# Data Directory

Place the four data files of each dataset in this folder. Below uses
`music4all_onion` as an example; for other datasets simply replace the dataset
name.

## Required files

| File | Format | Description |
|---|---|---|
| `music4all_onion.txt`        | one `user_id  item_id` pair per line, ordered chronologically per user | User–item interaction sequence. |
| `text_semantic_id.txt`       | `item_id \t c1,c2,c3` per line | 3-layer text-modality semantic IDs. |
| `audio_semantic_id.txt`      | `item_id \t c1,c2,c3` per line | 3-layer audio-modality semantic IDs. |
| `visual_semantic_id.txt`     | `item_id \t c1,c2,c3` per line | 3-layer visual-modality semantic IDs. |

## Notes

- `user_id` and `item_id` are 1-based integers.
- For datasets with fewer modalities (e.g. text only), simply omit the
  corresponding files; the model will skip the missing modality automatically.
- Pre-processed splits and semantic-ID files will be released together with
  the camera-ready paper.

## Expected directory layout

```
python/data/
├── README.md                  (this file)
├── music4all_onion.txt
├── text_semantic_id.txt
├── audio_semantic_id.txt
└── visual_semantic_id.txt
```
