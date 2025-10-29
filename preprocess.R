install.packages(c("MALDIquant", "MALDIquantForeign", "optparse"), repos = "https://cloud.r-project.org")
library("MALDIquant")
library("MALDIquantForeign")
library("optparse")

option_list <- list(
    make_option(c("-i", "--input_fp"), type="character", default=NULL,
                            help="Path to input CSV file", metavar="character"),
    make_option(c("-o", "--outdir"), type="character", default=NULL,
                            help="Output directory for preprocessed MALDI", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$outdir) || is.null(opt$input_fp)) {
    stop("Both --outdir and --input_fp must be provided.")
}

outdir <- opt$outdir
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("Created output directory: %s", outdir))
} else {
  message(sprintf("Output directory already exists: %s", outdir))
}

metadata_df <- read.csv(opt$input_fp)

for (row in 1:nrow(metadata_df)){
    tmp_tube_code <- metadata_df[row, "id"]
    tmp_dir <- metadata_df[row, "spot_dir"]
    tmp_fp <- file.path(tmp_dir, "1", "1Slin")
    
    # Load the data from fid
    spectra <- import(tmp_fp, removeEmptySpectra=T)
    
    # Check if length = 1
    if (length(spectra) == 1){
        spectra <- spectra[[1]]
        
        # Variance stabilization
        preprocessed_spectra <- transformIntensity(spectra, method='sqrt')
        
        # 4000-20000Da
        preprocessed_spectra  <- trim(preprocessed_spectra, range = c(4000,20000))
        
        # Savitzky-Golay-Filter smoothing
        preprocessed_spectra <- smoothIntensity(preprocessed_spectra, method='SavitzkyGolay', 
                                                                                        halfWindowSize=20)
        
        #  Baseline removal
        preprocessed_spectra <- removeBaseline(preprocessed_spectra, method="SNIP", iteration=40)
        
        preprocessed_spectra <- calibrateIntensity(preprocessed_spectra, method='median')
        
        
        mass_df <- data.frame(mass=mass(preprocessed_spectra),
                                                    intensity=intensity(preprocessed_spectra))
        
        write.csv(mass_df, file=file.path(outdir, paste(trimws(tmp_tube_code),".txt",sep="")), row.names=F)
    }
}
