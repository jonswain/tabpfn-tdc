```bash
git clone https://github.com/jonswain/tabpfn-tdc.git 
cd tabpfn-tdc
conda env create -f environment.yml
conda activate tabpfn-tdc
python submission.py | 2>&1 | tee -a log.txt
```