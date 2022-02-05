## Bucci Lab UMass Med
# vanni.bucci2@umassmed.edu

library("dada2")
library("ggplot2")
packageVersion("dada2")
library("phyloseq")
library("ggthemes")

#Create directory for files
mainDir <-  "../"
subDir <- "dada2_results"
ifelse(!dir.exists(file.path(mainDir, subDir)), dir.create(file.path(mainDir, subDir)), FALSE)
results_folder <- paste(mainDir,subDir,sep="")

#--------Filtering--------------------------------
# Cores to be used during filtering
cores <-  20

# Raw files directory
# files are downloaded from here: https://www.ebi.ac.uk/ena/browser/view/PRJEB14529?show=reads
raw_dir <-"" # CHANGE ME to the directory containing your demultiplexed fastq files

# Store Filtered file
path <- "./" # CHANGE ME to the directory where you want to store the trimmed data
filtpath <- file.path(path, "fastq") # Filtered files go into the filtered/ subdirectory
dir.create(filtpath,recursive = T)

# Only forward reads are used
#Forward reads
fil_fnFs <- sort(list.files(raw_dir, pattern=".fastq",
                        recursive = T,full.names = T))
# Full file name
pdf(paste0(results_folder,"/Forwards_fastq_before_fil.pdf"),width = 10,height = 10)
print(plotQualityProfile(fil_fnFs[1:8]))
dev.off()


# Filtered files
filtFs <- file.path(filtpath, basename(fil_fnFs))
# Forward reads only
out <- filterAndTrim(fil_fnFs, filtFs,
                     trimLeft= 12,
                     trimRight = 2,
                     # Removing initial nts to match previous seq length
                     maxEE=c(2) ,  rm.phix=TRUE,
                     compress=TRUE, verbose=TRUE, multithread=cores)


# Check the distribution of reads
library(ShortRead)
fastq_seq <- readFastq(filtFs[1])
table(nchar(fastq_seq@sread))

pdf(paste0(results_folder,"/Forwards_fastq_after_fil.pdf"),width = 10,height = 10)
plotQualityProfile(filtFs[1:8])
dev.off()

out <- data.frame(out)

# Extract sample names, assuming filenames have format: SAMPLENAME_XXX.fastq
sample.names <- sapply(strsplit(basename(filtFs), ".fastq"), `[`, 1)
names(filtFs) <- sample.names

set.seed(100)
# Learn forward error rates
errF <- learnErrors(filtFs,nbases =  3e8,multithread=cores,MAX_CONSIST = 20)

pdf(paste0(results_folder,"/Error_Forward.pdf"),width = 10,height = 10)
plotErrors(errF, nominalQ=TRUE)
dev.off()


# file with zero reads pass
f_name <-rownames(out)[out$reads.in == 0]

filtFs <-  filtFs[!filtFs %in% filtFs[grep(f_name,filtFs)]]

dadaFs <- dada(filtFs, err=errF, multithread=cores,pool = "pseudo")

save.image(paste0(path,"/After_dada2_pool_pseudo_run_1.RData"))

# Construct sequence table
seqtab <- makeSequenceTable(dadaFs)
dim(seqtab)

saveRDS(seqtab,paste0(results_folder,"/seqtab.rds")) 

# Inspect distribution of sequence lengths
table(nchar(getSequences(seqtab)))


set.seed(1079)
seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus",
                                    minFoldParentOverAbundance=2, 
                                    #minFoldParentOverAbundance=,
                                    multithread=cores,
                                    verbose=TRUE)
dim(seqtab.nochim)
sum(seqtab.nochim)/sum(seqtab)
table(nchar(getSequences(seqtab.nochim)))



out_fil <-  out[out$reads.in != 0,]
getN <- function(x) sum(getUniques(x))
track <- cbind(out_fil, sapply(dadaFs, getN),  rowSums(seqtab.nochim))
# If processing a single sample, remove the sapply calls: e.g. replace sapply(dadaFs, getN) with getN(dadaFs)
colnames(track) <- c("input", "filtered", "merged", "nonchim")
rownames(track) <- gsub(".fastq.*","",rownames(track))
head(track)
write.csv(track, paste0(results_folder,"/Reads_tracker.csv"))

# Write to disk
saveRDS(seqtab,paste0(results_folder,"/seqtab.rds")) # CHANGE ME to where you want sequence table saved
saveRDS(seqtab.nochim,paste0(results_folder,"/seqtab.nochim.rds")) # CHANGE ME to where you want sequence table saved
seq_variants <- colnames(seqtab.nochim) 
colnames(seqtab.nochim) <-  paste0("SV_",1:length(seq_variants))
seq_df <-  data.frame(t(seqtab.nochim))

write.csv(seqtab.nochim,paste0(results_folder,"/Counts_table.csv"))

#####Write a fasta file using inferred sequences
sequence <- seq_variants
names(sequence) <-rownames(seq_df)
library("Biostrings")
fasta_file <-  DNAStringSet(sequence)
writeXStringSet(fasta_file,paste0(results_folder,"/Seq_Var.fasta"))

## Assign taxonomy
## the SILVA DATABASESE ARE DOWNLOADED HERE https://zenodo.org/record/1172783#.YdLwib3MLXl
taxa <- assignTaxonomy(seq_variants, "../taxa_db/silva_nr_v132_train_set.fa.gz", multithread=cores)
taxa_sp <- addSpecies(taxa, "../taxa_db/silva_species_assignment_v132.fa.gz")

tax_tab <- taxa_sp
rownames(tax_tab) <- rownames(seq_df)
write.csv(tax_tab,paste0(results_folder,"/Tax_tab.csv"))

## Subset to use only the samples used in the MITRE paper ##
files_subset <-read.csv("../bokulich_filtered_subset.csv")
seqtab.nochim_import <- read.csv(paste0(results_folder,"/Counts_table.csv"))
tk <- which(seqtab.nochim_import$X %in% files_subset$x)
tk
seqtab.nochim_import_final <- seqtab.nochim_import[tk,]
dim(seqtab.nochim_import_final)
write.csv(seqtab.nochim_import_final,paste0(results_folder,"/abundance.csv"))

