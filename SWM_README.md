Order of operations:

* Get text file (cat all text files together)
* Shuffle (`shuffle_doc.sh`), with qlogin shell, or qsub it.
* `bash tok.sh`
* `bash mkvocab.sh`
* `bash data.sh`
* copy data to bucket (`copy_to_bucket.sh`)
* start TPU (`ctpu up --name=bert-trainer`)
* wait...
* check status (`ctpu status --name=bert-trainer`)
* delete TPU (`ctpu delete --name=bert-trainer`)