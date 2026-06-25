args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
png_path <- args[2]
top_n <- as.integer(args[3])

df <- read.csv(csv_path, encoding = "UTF-8", stringsAsFactors = FALSE)
df <- head(df, top_n)

freq_col <- ncol(df)
df[[freq_col]] <- as.numeric(df[[freq_col]])

if (any(is.na(df[[freq_col]]))) {
  stop("Frequency column contains non-numeric values")
}

col_name <- names(df)[1]

png(png_path, width = 1000, height = 700, res = 100)
par(mar = c(12, 5, 4, 2))
barplot(df[[freq_col]],
        names.arg = df[[col_name]],
        las = 2, col = "#4A90E2", border = NA,
        main = "頻出語句 (R)",
        ylab = "出現回数", cex.names = 0.9)
dev.off()
