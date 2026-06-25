args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
png_path <- args[2]
selected_meta <- args[3]

df <- read.csv(csv_path, fileEncoding = "UTF-8", row.names = 1, stringsAsFactors = FALSE)
df <- df[, colSums(df) > 0, drop = FALSE]
df <- df[rowSums(df) > 0, , drop = FALSE]

if (nrow(df) < 2 || ncol(df) < 2) {
  png(png_path, width = 600, height = 400)
  plot.new()
  text(0.5, 0.5, "Data insufficient for CA", cex = 1.5)
  dev.off()
  quit(save = "no", status = 1)
}

out_dir <- dirname(png_path)

ca_svd <- function(X) {
  total <- sum(X)
  P <- X / total
  r <- rowSums(P)
  c <- colSums(P)
  Z <- (P - outer(r, c)) / sqrt(outer(r, c))
  svd_res <- svd(Z, nu = 2, nv = 2)
  row_coords <- sweep(svd_res$u, 1, sqrt(r), "/")
  col_coords <- sweep(svd_res$v, 1, sqrt(c), "/")
  eigenvalues <- svd_res$d[1:2]^2
  list(row_coords = row_coords, col_coords = col_coords, eigenvalues = eigenvalues, total_inertia = sum(svd_res$d^2))
}

res <- ca_svd(as.matrix(df))
row_coords <- res$row_coords
col_coords <- res$col_coords
eigenvalues <- res$eigenvalues
total_inertia <- res$total_inertia

png(png_path, width = 1000, height = 800, res = 100)
par(mar = c(5, 5, 4, 2))
xlim <- range(c(row_coords[, 1], col_coords[, 1])) * 1.2
ylim <- range(c(row_coords[, 2], col_coords[, 2])) * 1.2
plot(row_coords[, 1], row_coords[, 2], type = "n",
     xlim = xlim, ylim = ylim,
     xlab = paste0("Dim 1 (", round(eigenvalues[1] / total_inertia * 100, 1), "%)"),
     ylab = paste0("Dim 2 (", round(eigenvalues[2] / total_inertia * 100, 1), "%)"),
     main = paste0("Correspondence Analysis: ", selected_meta))
abline(h = 0, v = 0, col = "gray", lty = "dashed")
points(row_coords[, 1], row_coords[, 2], col = "#4A90E2", pch = 16, cex = 1.2)
text(row_coords[, 1], row_coords[, 2], labels = rownames(df),
     col = "#333333", cex = 0.9, pos = 3)
points(col_coords[, 1], col_coords[, 2], col = "#E94A66", pch = 17, cex = 1.8)
text(col_coords[, 1], col_coords[, 2], labels = colnames(df),
     col = "#E94A66", cex = 1.2, pos = 3, font = 2)
legend("topright", legend = c("Words", selected_meta),
       col = c("#4A90E2", "#E94A66"), pch = c(16, 17), pt.cex = 1.5)
dev.off()

write.csv(data.frame(Word = rownames(df), Dim1 = row_coords[, 1], Dim2 = row_coords[, 2],
                     stringsAsFactors = FALSE),
          file = file.path(out_dir, "ca_row_coords.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(data.frame(Meta = colnames(df), Dim1 = col_coords[, 1], Dim2 = col_coords[, 2],
                     stringsAsFactors = FALSE),
          file = file.path(out_dir, "ca_col_coords.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(data.frame(Dimension = 1:2, Eigenvalue = eigenvalues,
                     Contribution_pct = eigenvalues / total_inertia * 100,
                     stringsAsFactors = FALSE),
          file = file.path(out_dir, "ca_eigenvalue.csv"), row.names = FALSE, fileEncoding = "UTF-8")
