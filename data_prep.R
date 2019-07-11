# ------------------------------------------------------------------------ #
#
# Give me some credit - gagnavinnsla
#
# ------------------------------------------------------------------------ #


library(tidyverse)
library(ComplexHeatmap)       # Til að visualize-a missing values
library(visdat)               # Til að visualize-a missing values
library(naniar)               # Til að visualize-a missing values
library(tidymodels)


df <- read_csv("cs-training.csv") %>% select(-X1) %>% janitor::clean_names()
df_test <- read_csv("cs-test.csv") %>% select(-X1) %>% janitor::clean_names()

urtak <- sample(150000, 1000, replace = FALSE)
df_sub <- df[urtak, ]
vis_miss(df_sub)


convert_missing <- function(x) ifelse(is.na(x), 0, 1)

gogn <- df

df_missing <- apply(gogn, 2, convert_missing)

Heatmap(
            df_missing,
            name = "Missing",
            column_title = "Predictors", row_title = "Samples",
            col = c("black", "lightgrey"),
            show_heatmap_legend = FALSE,
            row_names_gp = gpar(fontsize = 0)
)


gg_miss_upset(df)



# PCA ---------------------------------------------------------------------

missing_ppn <-
            apply(df, MARGIN = 1, function(x)
                        sum(is.na(x)))/ncol(df)


pca_df <- prcomp(df_missing)


pca_d <- data.frame(pca_df$x) %>%
            select(PC1, PC2) %>%
            mutate(Percent = missing_ppn*100) %>%
            distinct(PC1, PC2, Percent)

pca_d_rng <- extendrange(c(pca_d$PC1, pca_d$PC2))

ggplot(pca_d,
       aes(x = PC1, y = PC2, size = Percent)) +
            geom_point(alpha = 0.5) +
            xlim(pca_d_rng) +
            ylim(pca_d_rng)



biplot(pca_d, scale = 0)   # Sjá bls. 401 í ITSL PCA lab




# Skoða betur monthly income ----------------------------------------------

df_miss_income <- df %>%
            mutate(miss_income = as.character(case_when(is.na(monthly_income) ~ 1,
                                           TRUE ~ 0)))


# Relation to Serious_dlqin2yrs

graf_monthly <- function(df = df_miss_income, x = df_miss_income$miss_income, y) {
            ggplot(df,
                   aes(x = x,
                       y = y)) +
                        geom_boxplot(outlier.shape = NA) +
                        scale_y_continuous(limits = quantile(y, c(0.1, 0.9)))
}


graf_monthly(y = df_miss_income$revolving_utilization_of_unsecured_lines)
graf_monthly(y = df_miss_income$age)
graf_monthly(y = df_miss_income$debt_ratio) # Sterkt samband hér, gæti hjálpað til við að spá fyrir um monthly income




# Skoða betur number of dependencies --------------------------------------

df_miss_dep <- df %>%
            mutate(miss_dep = as.character(case_when(is.na(number_of_dependents) ~ 1,
                                        TRUE ~ 0)))

graf_depend <- function(df = df_miss_dep, x = df_miss_dep$miss_dep, y) {
            ggplot(df,
                   aes(x = x,
                       y = y)) +
                        geom_boxplot(outlier.shape = NA) +
                        scale_y_continuous(limits = quantile(y, c(0.1, 0.9)))
}

df_miss_dep %>% group_by(miss_dep) %>% summarise(hlutfall = mean(serious_dlqin2yrs))

graf_depend(y = df_miss_dep$revolving_utilization_of_unsecured_lines)
graf_depend(y = df_miss_dep$age)
graf_depend(y = df_miss_dep$debt_ratio)
df_miss_dep %>% group_by(miss_dep) %>% summarise(debt = mean(debt_ratio)) # Mun hærra
df_miss_dep %>% group_by(miss_dep) %>% summarise(hlutfall = mean(serious_dlqin2yrs)) # nokkuð líkt




# Impute-a missing values -------------------------------------------------

vapply(df, function(x) mean(!is.na(x)), numeric(1))


# initial recipe

rec_obj <- recipe(serious_dlqin2yrs ~ ., data = df)

imputed_knn <- rec_obj %>%
            step_knnimpute(all_predictors())

imputed_bag <- rec_obj %>%
            step_bagimpute(all_predictors())


standardize_knn <- imputed_knn %>%
            step_center(all_predictors()) %>%
            step_scale(all_predictors())


trained_rec_knn <- prep(standardize_knn, training = df)
trained_rec_bag <- prep(imputed_bag, training = df)


# KNn gögn
train_data_knn <- bake(trained_rec_knn, new_data = df)
test_data_knn <- bake(trained_rec_knn, new_data = df_test)


# bagging gögn
train_data_bag <- bake(trained_rec_bag, new_data = df)
test_data_bag <- bake(trained_rec_bag, new_data = df_test)


# Vista gögnin
write_csv(train_data_knn, "train_knn.csv")
write_csv(test_data_knn, "test_knn.csv")
write_csv(train_data_bag, "train_bag.csv")
write_csv(test_data_bag, "test_bag.csv")
