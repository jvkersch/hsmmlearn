#!/bin/sh

# Install the hsmm R package via CRAN

cat <<EOF > "$HOME/.Rprofile"
r = getOption("repos") # hard code the UK repo for CRAN
r["CRAN"] = "http://cran.uk.r-project.org"
options(repos = r)
rm(r)
EOF

echo 'install.packages("hsmm")' > cmd.R
Rscript cmd.R
