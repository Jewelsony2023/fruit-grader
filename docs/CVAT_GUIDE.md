# CVAT guide: fixing the test-set masks

**Goal:** we have 227 test photos with *computer-made* (pseudo) masks. They are often wrong. Your job is to correct them so they become the **ground truth** that the final model is judged against. Accuracy matters more than speed, but most images only need small fixes.

You get **one batch** (about 76 images): `batch_1`, `batch_2` or `batch_3`.

---

## 1. Set up CVAT (once, about 30 min)

1. Install **Docker Desktop** for Windows (https://www.docker.com/products/docker-desktop/). Accept the WSL 2 option it suggests and restart if asked. Start Docker Desktop and wait until it says "Engine running".
2. Install **Git** (https://git-scm.com/download/win) if you don't have it.
3. Open **PowerShell** and run:
   ```powershell
   cd $HOME
   git clone https://github.com/cvat-ai/cvat
   cd cvat
   docker compose up -d
   ```
   The first start downloads a few GB and takes several minutes.
4. Create your login (answer the questions; email can be blank):
   ```powershell
   docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
   ```
5. Open **http://localhost:8080** in Chrome and log in.

To stop CVAT later: `docker compose down` in the `cvat` folder. **Never** use `docker compose down -v`, which deletes your work. Start it again with `docker compose up -d`.

---

## 2. Create your task

1. You received two files: `batch_<k>_images.zip` and `batch_<k>_annotations.zip`.
2. In CVAT: **Tasks → + (Create a new task)**.
3. **Name:** `batch_<k>` (for example `batch_2`).
4. **Labels:** click **Raw** and replace the content with exactly this, then click **Done**:
   ```json
   [
     {"name": "fruit",   "color": "#33ddff", "type": "polygon", "attributes": []},
     {"name": "bruise",  "color": "#9b59b6", "type": "polygon", "attributes": []},
     {"name": "rot",     "color": "#e02828", "type": "polygon", "attributes": []},
     {"name": "cut",     "color": "#f0b414", "type": "polygon", "attributes": []},
     {"name": "healthy", "color": "#3caa5a", "type": "polygon", "attributes": []}
   ]
   ```
   The names must be spelled exactly like this, all lowercase.
5. **Select files:** upload `batch_<k>_images.zip`.
6. Click **Submit & Open**. Wait until the task shows a job.

## 3. Load the pre-made masks

1. On the task page: **Actions → Upload annotations**.
2. **Import format:** `COCO 1.0`. **File:** `batch_<k>_annotations.zip`. Confirm.
3. Open the job (click **Job #...**). You should see coloured polygons on the first image (`t###.jpg`).

If the upload fails, check that the five label names match exactly (step 2.4).

---

## 4. What each label means

Annotate **only the main fruit** (the biggest, most central one). Other fruit, half-fruit at the edge, cut slices and leaves are **background**: no polygon.

| Label | What it is | Typical look |
|---|---|---|
| **fruit** | The whole outline of the main fruit, **defects included** | One polygon around the entire fruit skin |
| **bruise** | Skin intact, flesh damaged underneath | Smooth, soft-edged darker patch (brownish, purplish, dull); no mould, not sunken or wrinkled |
| **rot** | Decay | Brown/black, sunken, wrinkled or leathery area; **any mould** (white, grey, green or blue fuzz); oozing; black decayed banana skin |
| **cut** | Mechanical damage: skin broken by force | Cuts, punctures, cracks, scratches, bites, often with sharp edges and pale flesh showing |
| **healthy** | Healthy skin completely **surrounded** by a defect | Only needed for an island inside a ring of rot/bruise (see 5.4) |

**Bruise or rot?** If there is mould, or the skin is broken, wrinkled or sunken: **rot**. If the skin is smooth and intact and only discoloured: **bruise**. If you still can't decide: **rot**.

### NOT a defect (it stays plain "fruit", no defect polygon)
- **Stem** and **calyx** (the dried star/flower end at the bottom of an apple or orange; the dark tips at both ends of a banana)
- **Natural colour:** red streaks, a deep-red side, yellow-green patches, an orange's slightly greener area, a banana's green neck
- **Lenticels:** the tiny light dots on apple skin
- **Banana natural speckles:** small brown "sugar spots" on a ripe yellow banana. A spot becomes **rot** only when it merges into large dark, sunken or mushy areas
- **Stickers and labels:** leave them inside the fruit outline, with no defect polygon
- **Shine:** highlights, reflections, water droplets
- **Shadows:** a shadow on the table is background. Natural darkening toward the fruit's edge is fruit, not a bruise
- **Orange peel texture** (the pores)
- **Leaves:** background, so remove them from the fruit outline

---

## 5. How to fix polygons fast

Work image by image. For each one:

1. **Fruit outline first.** Does the `fruit` polygon follow the fruit's edge? Common mistakes: it misses a white mouldy side, includes a leaf or shadow, or covers two fruits.
2. **Delete wrong defects.** Click the polygon and press **Delete**. Typical wrong ones: stem or calyx marked as rot, natural deep-red skin, shine.
3. **Fix the type.** If a polygon is right but the label is wrong (for example bruise but it's really rot), click it and change the label in the object list on the right.
4. **Fix the shape** (see the tips below), and **add missing defects** (white/green mould is often missed).
5. **Save** (**Ctrl+S**) and go to the next image (**F**; **D** goes back).

Tips:
- **Move a point:** drag it. **Delete a point:** right-click the point and choose *Delete point*.
- **Redraw part of an outline:** hold **Shift** and click a point of the outline where the wrong part starts, click your new points, and finish by clicking a point where the wrong part ends.
- Shortcuts can differ slightly between CVAT versions. Hover over a button to see its shortcut, or press **F1** for the full list.
- **Draw a new polygon:** choose the polygon tool on the left, pick the label, click the points, and press **N** to finish. Pressing **N** again starts another polygon with the same tool.
- **Undo:** Ctrl+Z.
- Polygons **may overlap**: defects always sit on top of the fruit outline, so you never need to cut holes in the fruit polygon.
- A defect polygon cannot have a hole. If healthy skin is surrounded by a defect, either split the defect into two polygons around it, or draw a **healthy** polygon over the island.
- Small defects matter. Zoom in with the mouse wheel. Don't spend more than about 3 minutes on one image.
- If a photo is unusable (not a fruit, the fruit can't be seen), leave the polygons as they are and write the file name (for example `t045.jpg`) in your notes for the project owner.

Please don't try to guess whether the fruit was labelled "fresh" or "rotten" originally, and don't open `manifest.csv`. Judge only what you see.

---

## 6. Export and send back

1. Go back to the task page: **Actions → Export task dataset**.
2. **Format:** `COCO 1.0`. Leave **Save images** **unchecked**.
3. The download is a zip. Rename it **`batch_<k>_<your initials>_coco.zip`** (for example `batch_2_AK_coco.zip`).
4. Send it to the project owner together with your notes (unusable images, doubts).

Don't rename, add or remove images, and don't change the label names.

---

## For the project owner: converting a returned batch

```powershell
# unzip the returned file into data/cvat_returned/batch_<k>/ (it contains annotations/instances_default.json)
python -m fruitgrade.data.coco_to_masks `
  --json data/cvat_returned/batch_<k>/annotations/instances_default.json `
  --images data/cvat/batch_<k>/images `
  --manifest data/cvat/manifest.csv `
  --map "fruit=healthy,bruise=bruise,rot=rot,cut=mechanical,healthy=healthy_patch" `
  --prefix cvat
```
This appends the corrected masks to `data/processed/labels.csv` as `mask_quality=verified` test rows, keeping each photo's group and source path. The export is made with `python -m fruitgrade.data.export_for_cvat`; a tracked copy of the manifest is in `results/cvat/manifest.csv`.
