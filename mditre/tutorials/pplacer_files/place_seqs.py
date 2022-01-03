import sys
import os
import shutil
import re
import argparse
from subprocess import call
import time

from Bio import SeqIO, Phylo
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import ete3

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--v4-region-start', type=int, dest='v4_region_start',
        help='Position of the start (inclusive) of the v4 region in the 16S gene of the ' \
            'aligned reference sequences')
    parser.add_argument('--v4-region-end', type=int, dest='v4_region_end',
        help='Position of the end (exclusive) of the v4 region in the 16S gene of the ' \
            'aligned reference sequences')
    parser.add_argument('--refpkg', type=str, dest='refpkg',
        help='Folder of the refpkg that you built (.refpkg)')
    parser.add_argument('--query-reads', type=str, dest='query_reads',
        help='These are the sequences we want to place')
    parser.add_argument('--verbose', type=int, dest='verbose', default=0,
        help='Print out the commands')
    parser.add_argument('--temp-folder', type=str, dest='temp_folder', 
        default='tmp/', help='Temporary folder')
    parser.add_argument('--output-folder', '-o', type=str, dest='output_folder', 
        default='output/', help='Output folder')
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    verbose = bool(args.verbose)
    temp_folder = args.temp_folder
    if temp_folder[-1] != '/':
        temp_folder += '/'
    output_folder = args.output_folder
    if output_folder[-1] != '/':
        output_folder += '/'

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    

    refpkg = args.refpkg
    if refpkg[-1] != '/':
        refpkg += '/'

    # Extract the v4 region
    # ---------------------
    print('\n\nExtract v4 region')
    fname_v4_aligned_refpkg = temp_folder + 'aligned_reference_sequences_v4_region.fa'
    start = args.v4_region_start
    stop = args.v4_region_end

    refpkg_alignment = refpkg + 'RDP-11-5_TS_Processed_Aln.fa'

    seqs = SeqIO.to_dict(SeqIO.parse(refpkg_alignment, 'fasta'))
    trimmed = []
    for k,record in seqs.items():
        record.seq = record.seq[start:stop]
        trimmed.append(record)
    SeqIO.write(trimmed, fname_v4_aligned_refpkg, 'fasta')

    # Create HMM for aligned v4 region
    # --------------------------------
    print('\n\nCreate HMM for v4 region')
    hmm_model = temp_folder + 'aligned_reference_sequences_v4_region.hmm'
    command = 'hmmbuild ' + hmm_model + ' ' + fname_v4_aligned_refpkg
    if verbose:
        print(command)
    os.system(command)

    # Align the sequences to the reference package v4 region
    # ------------------------------------------------------
    print('\n\nAlign sequences to reference package')
    align_output_v4 = output_folder + 'placed_sequences_on_v4_region.sto'
    query_reads = args.query_reads
    command = 'hmmalign --trim --mapali ' + fname_v4_aligned_refpkg + \
        ' -o ' + align_output_v4 + ' ' + hmm_model + ' ' + query_reads
    if verbose:
        print(command)
    os.system(command)

    # Place aligned sequences on the reference tree
    # ---------------------------------------------
    print('\n\nPlace aligned sequences on the reference tree')
    pplacer_output = output_folder + 'placement.jplace'
    command = 'pplacer --verbosity 1 -c ' + refpkg + ' ' + align_output_v4 + \
        ' -o ' + pplacer_output
    if verbose:
        print(command)
    os.system(command)

    # Place on XML tree
    # -----------------
    print('\n\nGet placement on xml tree')
    xml_output = temp_folder + 'xml_tree.xml'
    command = 'guppy tog -o ' + xml_output + ' --xml ' + pplacer_output
    if verbose:
        print(command)
    os.system(command)

    # Make tree with taxonomic name
    # -----------------------------
    print('\n\nReplace species id with taxonomy for references')
    species_output = output_folder + 'taxonomy.xml'
    refpkg_species_info = refpkg + 'RDP-11-5_TS_Processed_seq_info.csv'
    command = 'scripts/replace_ID_taxaName.pl ' + xml_output + ' ' + refpkg_species_info
    if verbose:
        print(command)
    os.system(command)

    # Make the newick trees
    # ---------------------
    print('\n\nMaking newick tree')
    newick_output = output_folder + 'newick_tree_full.nhx'

    Phylo.convert(xml_output, 'phyloxml', newick_output, 'newick')
    tree = Phylo.read(xml_output, 'phyloxml')

    print('Trim to only query reads')
    seqs = SeqIO.parse(query_reads, 'fasta')
    seqs = SeqIO.to_dict(seqs)
    asvs = list(seqs.keys())
    tree = ete3.Tree(newick_output)
    tree.write(format=1, outfile=newick_output)
    print('pruning')
    tree.prune(asvs, preserve_branch_length=True)
    tree.write(format=1, outfile=output_folder + 'newick_tree_query_reads.nhx')







