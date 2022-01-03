pip install biopython

conda install -c bioconda hmmer pplacer

python place_seqs.py --refpkg ./RDP-11-5_TS_Processed.refpkg --query-reads Seq_Var.fasta --verbose 1