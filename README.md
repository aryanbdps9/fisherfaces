# fisherfaces
CS663 project
## Description of various files:
- [Yale face dataset in png format](https://cseiitbacin-my.sharepoint.com/:f:/g/personal/aryan_cse_iitb_ac_in/Et6BnVVe5F5NjplmIFJL3WIB-BxYNXQNtb3E0or9-7dkMg?e=0iIo1x) (download and extract this folder with the name pngyalefaces)
- `helper.py`: contains helper functions which are useful in things like loading data, etc.
- `corefn.py`: The algorithms will be implemented here. This is just like `myMainScript.py` of assignments.
- [Large Yale face dataset in png format](https://cseiitbacin-my.sharepoint.com/:f:/g/personal/aryan_cse_iitb_ac_in/EtswDEWht01Bq1GdRjRIOAABizoB696okFiLIQTusce02w?e=h4BR90)
- `pgmread.py`: python script to convert images of any format to png recursively. Just give the path of folder containing images and the path of the destination folder in the function `change_img_ext_rec`
- `PNGExtendedYale.tar.gz`: Extended Yale dataset. We will use it later(as it is large, so, longer training/testing time). Currently not uploaded due to large size.

**Warning: Do NOT add folders containing images into the repo. git will become very slow. Add the folders to the end of .gitignore instead**
