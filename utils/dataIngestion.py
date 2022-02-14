import pandas as pd


# Data Ingestion
def readDataFrame(file):
    # Create Data Frame
    df = pd.read_csv(file, delimiter='\t', dtype=object)

    # Replace Null and NaN into empty string
    df.fillna("", inplace=True)

    # col labels
    df.columns = [
        'id',
        'label',
        'statement',
        'subjects',
        'aspect',
        'speaker',
        'speaker_job_title',
        'state_info',
        'party_affiliation',

        'barely_true',
        'false',
        'half_true',
        'mostly_true',
        'pants_on_fire',
        'true',

        'context'
    ]

    return df
