# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "seaborn",
#   "python-dotenv",
#   "requests",
# ]
# ///

#!/usr/bin/env python3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from dotenv import load_dotenv
import requests

# Load environment variables from a .env file (such as API keys)
load_dotenv()

# Retrieve API key for external API calls
api_key = os.environ.get("AIPROXY_TOKEN")

# Function to load a CSV file and handle encoding errors
def load_csv(file_path):
    try:
        # Try default UTF-8 encoding first
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully with UTF-8 encoding.")
        return df
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying 'latin1' encoding...")
        try:
            # Fallback to 'latin1' encoding
            df = pd.read_csv(file_path, encoding='latin1')
            print("CSV file loaded successfully with 'latin1' encoding.")
            return df
        except Exception as e:
            print(f"Error loading file with 'latin1' encoding: {e}")
            exit()
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

# Function to classify columns into categories (ID, numerical, categorical, date)
def classify_columns(df):
    # Columns that contain 'id' or 'code'
    id_cols = [col for col in df.columns if "id" in col.lower() or "code" in col.lower()]
    # Numerical columns excluding id columns
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns.difference(id_cols)
    # Categorical columns (strings)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # Date columns (attempt to parse datetime)
    date_cols = df.select_dtypes(include=['datetime', 'object']).apply(lambda col: pd.to_datetime(col, errors='coerce')).dropna(axis=1).columns
    return id_cols, numerical_cols, categorical_cols, date_cols

# Function to analyze dataset and generate insights
def analyze_data(df):
    # Classify columns and generate an overview of the dataset
    id_cols, numerical_cols, categorical_cols, date_cols = classify_columns(df)

    insights = {
        "overview": {
            "num_rows": df.shape[0],  # Number of rows
            "num_cols": df.shape[1],  # Number of columns
            "missing_values": df.isnull().sum().to_dict()  # Missing values per column
        },
        "numerical_summary": df[numerical_cols].describe().to_dict(),  # Numerical column stats
        "categorical_summary": {col: df[col].value_counts().to_dict() for col in categorical_cols}  # Categorical value counts
    }

    return insights

# Function to summarize the insights for easier storytelling
def summarize_insights(insights):
    summarized_insights = {
        "overview": insights["overview"],
        # Limit the numerical summary to only 3 columns
        "numerical_summary": {key: insights["numerical_summary"][key] for key in list(insights["numerical_summary"])[:3]},
        # Limit the categorical summary to only 2 columns with top 5 values each
        "categorical_summary": {key: list(insights["categorical_summary"][key].keys())[:5] for key in list(insights["categorical_summary"])[:2]}
    }
    return summarized_insights

# Function to generate a story based on summarized insights
def generate_story(insights, csv_name):
    summarized_insights = summarize_insights(insights)

    prompt = (
        f"You are a data storyteller. Analyze the following summarized insights from a dataset named '{csv_name}':\n\n"
        f"Overview: {summarized_insights['overview']}\n\n"
        f"Numerical Summary: {summarized_insights['numerical_summary']}\n\n"
        f"Categorical Summary: {summarized_insights['categorical_summary']}\n\n"
        f"Write a brief and engaging story, no longer than 700 words, summarizing the dataset and its potential meaning."
    )

    try:
        # Make a POST request to an external API (e.g., GPT-based model) to generate the story
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'  # Include the API key in the request headers
        }
        response = requests.post(r"http://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, 
                                 json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a creative storyteller."},
                                                                             {"role": "user", "content": prompt}]})
        response = response.json()
        return response['choices'][0]['message']['content']  # Extract the story content from the response
    except Exception as e:
        print(f"Error generating story: {e}")
        return "Story generation failed."

# Function to save the generated story and visualizations to files
def save_story_and_images(csv_path, insights, story):
    # Create a folder using the CSV file name (without extension)
    csv_name = Path(csv_path).stem
    folder_path = Path(csv_name)
    folder_path.mkdir(exist_ok=True)  # Create folder if it doesn't exist

    # Save the generated story in a README.md file
    readme_path = folder_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# Story for {csv_name}\n\n")
        f.write(story)

    # Generate and save visualizations for numerical columns
    numerical_cols = pd.DataFrame(insights['numerical_summary'])
    for col in numerical_cols.columns:
        plt.figure()
        sns.histplot(numerical_cols[col], kde=True)  # Plot distribution of each numerical column
        img_path = folder_path / f"{col}_distribution.png"
        plt.savefig(img_path)  # Save the plot as an image file
        plt.close()  # Close the plot

        # Add image reference in the README file
        with open(readme_path, "a") as f:
            f.write(f"![{col} Distribution](./{col}_distribution.png)\n")

    print(f"Story and images saved in folder: {folder_path}")

# Main function to execute the script
def main():
    # Parse command-line arguments to get the CSV file path
    parser = argparse.ArgumentParser(description="Autolysis: Automatic Data Analysis and Storytelling")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to analyze")
    args = parser.parse_args()

    # Load the CSV file and process it
    csv_path = args.csv_path
    df = load_csv(csv_path)
    insights = analyze_data(df)
    csv_name = Path(csv_path).stem  # Extract CSV name (without extension)
    story = generate_story(insights, csv_name)  # Generate a story based on insights
    save_story_and_images(csv_path, insights, story)  # Save the story and visualizations

# Entry point of the script
if __name__ == "__main__":
    main()
