from modules.data_preprocessors import transpose_to_all_keys, get_chord_name
import itertools
import pandas as pd
import os
import random
import numpy as np


def create_phrase_location_list(csv_path, pll_path):
    def phrase_to_string(begin_phrase_index, end_phrase_index):
        return "{}:{}".format(begin_phrase_index, end_phrase_index)

    def movement_to_string(movement_name, phrase_list):
        return "{} {}".format(movement_name, " ".join(phrase_list))

    def get_movement_name(no, mov):
        return "{:02d}_{}".format(no, mov)

    df = pd.read_csv(csv_path)
    movement_name = get_movement_name(df["no"][0], df["mov"][0])
    movement_list = []
    phrase_list = []
    begin_phrase_index = 0
    end_phrase_index = 0
    last_index = len(df) - 1

    for i, row in df.iterrows():
        if row["phraseend"]:
            end_phrase_index = i + 1
            phrase_list.append(phrase_to_string(begin_phrase_index, end_phrase_index))
            begin_phrase_index = end_phrase_index

        if i == last_index:
            if end_phrase_index != i + 1:
                end_phrase_index = i + 1
                phrase_list.append(
                    phrase_to_string(begin_phrase_index, end_phrase_index)
                )
                begin_phrase_index = end_phrase_index

            movement_list.append(movement_to_string(movement_name, phrase_list))
            phrase_list = []
            break

        current_movement_name = get_movement_name(row["no"], row["mov"])
        if current_movement_name != movement_name:
            if end_phrase_index != i:  # end of movement but not end of phrase
                end_phrase_index = i
                phrase_list.append(
                    phrase_to_string(begin_phrase_index, end_phrase_index)
                )
                begin_phrase_index = end_phrase_index

            movement_list.append(movement_to_string(movement_name, phrase_list))
            phrase_list = []
            movement_name = current_movement_name

    with open(pll_path, "w") as f:
        f.write("\n".join(movement_list))


def get_dataframe_from_locations(df, phrase_list, augment_func):
    original_df_list = []
    augmented_df_list = []

    for begin_idx, end_idx in phrase_list:
        original_df = df[begin_idx:end_idx]
        original_df_list.append(original_df)

        augmented_dfs = augment_func(original_df)
        augmented_df_list += augmented_dfs

    return original_df_list, augmented_df_list


def convert_row_to_chord_name(row):
    if pd.isnull(row["numeral"]):
        return np.nan

    global_key = row["global_key"]
    local_key = row["local_key"]
    numeral = row["numeral"]
    form = row["form"] if not pd.isnull(row["form"]) else None
    figbass = str(int(row["figbass"])) if not pd.isnull(row["figbass"]) else None
    relativeroot = row["relativeroot"] if not pd.isnull(row["relativeroot"]) else None
    # changes = str(row["changes"]) if not pd.isnull(row["changes"]) else None

    name = get_chord_name(
        global_key, local_key, numeral, form, figbass, relativeroot, True
    )
    return name


def convert_to_chord_name_progression_string(df):
    progression = df.apply(convert_row_to_chord_name, axis=1)
    return [chord for chord in progression if not pd.isnull(chord)]


def skip_repetitions(chord_list, skip_repetitions=True, skip_double_repetitions=True):
    if skip_repetitions:
        chord_list = [k for k, g in itertools.groupby(chord_list)]

    if skip_double_repetitions:
        n = len(chord_list)
        i = 0
        tmp_chord_list = []
        while i < n:
            if 1 < i and i + 1 < n:
                if (
                    chord_list[i - 2] == chord_list[i]
                    and chord_list[i - 1] == chord_list[i + 1]
                ):
                    i += 2
                    continue
                else:
                    tmp_chord_list.append(chord_list[i])
                    i += 1
            else:
                tmp_chord_list.append(chord_list[i])
                i += 1
        chord_list = tmp_chord_list
    return chord_list


def get_progression_list_from_dataframe_list(df_list, transform_func):
    progression_list = []
    for df in df_list:
        progression = transform_func(df)
        progression = skip_repetitions(progression)
        if len(progression) > 1:
            progression_list.append(" ".join(progression))
    return progression_list


def split_into_k_folds(csv_path, pll_path, folds_path, k, shuffle=True):
    def write_progression_list_to_file(progression_list, file_name):
        file_path = os.path.join(folds_path, file_name)
        with open(file_path, "w") as f:
            f.write("\n".join(progression_list))

    df_all = pd.read_csv(csv_path)

    with open(pll_path) as f:
        movement_list = f.read().split("\n")

    all_phrase_list = []
    for movement in movement_list:
        phrase_list_str = movement.split(" ")[1:]
        for phrase_str in phrase_list_str:
            begin_idx, end_idx = phrase_str.split(":")
            begin_idx, end_idx = int(begin_idx), int(end_idx)
            all_phrase_list.append((begin_idx, end_idx))

    if shuffle:
        random.shuffle(all_phrase_list)

    num_phrases = len(all_phrase_list)
    phrases_per_fold = num_phrases // k + 1
    for i, begin in enumerate(range(0, num_phrases, phrases_per_fold)):
        fold_phrase_list = all_phrase_list[begin : begin + phrases_per_fold]
        fold_original_df_list, fold_augmented_df_list = get_dataframe_from_locations(
            df_all, fold_phrase_list, transpose_to_all_keys
        )

        fold_original_progression_list = get_progression_list_from_dataframe_list(
            fold_original_df_list, convert_to_chord_name_progression_string
        )
        fold_augmented_progression_list = get_progression_list_from_dataframe_list(
            fold_augmented_df_list, convert_to_chord_name_progression_string
        )

        write_progression_list_to_file(
            fold_original_progression_list, "original_fold_{}.txt".format(i)
        )
        write_progression_list_to_file(
            fold_augmented_progression_list, "augmented_fold_{}.txt".format(i)
        )


def join_folds(folds_path, indices, augment):
    dataset = []
    for index in indices:
        file_path = os.path.join(
            folds_path,
            "{}_fold_{}.txt".format("augmented" if augment else "original", index),
        )
        with open(file_path) as f:
            dataset += f.read().split("\n")
    return dataset


def write_dataset_to_file(cv_path, name, dataset):
    file_path = os.path.join(cv_path, "{}.txt".format(name))
    with open(file_path, "w") as f:
        f.write("\n".join(dataset))


def create_cross_validation_datasets(folds_path, cv_path, k, ratio):
    num_folds = k // ratio[2]
    for i in range(num_folds):
        train = join_folds(
            folds_path, [(idx + i) % k for idx in range(ratio[0])], augment=True
        )
        val = join_folds(
            folds_path, [(idx + i) % k for idx in range(ratio[1])], augment=False
        )
        test = join_folds(
            folds_path, [(idx + i) % k for idx in range(ratio[2])], augment=False
        )

        saved_path = os.path.join(cv_path, str(i))
        if not os.path.isdir(saved_path):
            os.makedirs(saved_path)

        write_dataset_to_file(saved_path, "train", train)
        write_dataset_to_file(saved_path, "val", val)
        write_dataset_to_file(saved_path, "test", test)


CSV_PATH = "data/all_annotations.csv"
PLL_PATH = "data/phrase_location_list.txt"
FOLDS_PATH = "data/folds/"
CV_PATH = "data/cv/"
K = 10

if not os.path.isdir(FOLDS_PATH):
    os.makedirs(FOLDS_PATH)

create_phrase_location_list(CSV_PATH, PLL_PATH)
split_into_k_folds(CSV_PATH, PLL_PATH, FOLDS_PATH, K)
create_cross_validation_datasets(FOLDS_PATH, CV_PATH, K, ratio=[7, 1, 2])

