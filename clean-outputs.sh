#!/bin/bash

# Set the name of your text file here
FOLDER_LIST_FILE="folders_to_clean.txt"
LOG_FILE="deletion_log.txt"

echo "Folder Deletion Log - $(date)" > "$LOG_FILE"
echo "----------------------------------------------------" >> "$LOG_FILE"

# Check if the folder list file exists
if [ ! -f "$FOLDER_LIST_FILE" ]; then
    echo "ERROR: Folder list file '$FOLDER_LIST_FILE' not found!"
    exit 1
fi

# Read the file line by line
while IFS= read -r folder_path || [ -n "$folder_path" ]; do
    
    # Remove any leading/trailing whitespace
    folder_path=$(echo "$folder_path" | xargs)
    
    # Skip empty lines
    if [ -z "$folder_path" ]; then
        continue
    fi
    
    echo "Attempting to delete: $folder_path"
    
    # Check if the folder exists
    if [ -d "$folder_path" ]; then
        # *** The -r switch is crucial here: it deletes directories and their contents recursively
        # *** The -f switch forces the deletion (no prompts)
        rm -rf "$folder_path"
        
        if [ -d "$folder_path" ]; then
            echo "ERROR: Failed to delete $folder_path" >> "$LOG_FILE"
            echo "FAILED: $folder_path"
        else
            echo "SUCCESS: Deleted $folder_path" >> "$LOG_FILE"
            echo "DELETED: $folder_path"
        fi
    else
        echo "WARNING: Path not found - $folder_path" >> "$LOG_FILE"
        echo "NOT FOUND: $folder_path"
    fi

done < "$FOLDER_LIST_FILE"

echo ""
echo "Deletion process complete."
echo "Check '$LOG_FILE' for details."