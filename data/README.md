# UEA multivariate data

Download the aeon `.ts` version of the UEA multivariate archive:

~~~bash
bash data/download_uea.sh
export UEA_ROOT="$PWD/data/UEA"
~~~

Direct download:

https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip

Dataset page:

https://www.timeseriesclassification.com/dataset.php

The archive is ignored by Git. The experiment scripts expect:

    $UEA_ROOT/<Dataset>/<Dataset>_TRAIN.ts
    $UEA_ROOT/<Dataset>/<Dataset>_TEST.ts