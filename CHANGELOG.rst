==========================
mitarspysigproc Change Log
==========================

.. current developments

v1.0.1
====================

**Added:**

* sub_integration function added to first arithmatically average and then use meadian to average.

**Changed:**

* ACF estimators work across n-dimensional arrays where only the first two dimensions will be used in the estimation and rest will be passed through.

**Fixed:**

* Errors in the acf estimator causing failures has been fixed.



v1.0.0
====================

**Added:**

* Documentation via sphinx and readthedocs
* ACF estimation for noncoherent processing
* Conda build
* pypi build

**Changed:**

* Chirp and sine test are now just test code for the PFBs.
* Moved scripts with plotting to examples
* Examples are also located in doc/source/notebooks in the form of jupyter notebooks.


