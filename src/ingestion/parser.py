import re
from bs4 import BeautifulSoup


def extract_pure(raw_submission: str) -> str:
    """
    Isolates the primary 10-K document from a full SEC submission.

    Filters the raw content to extract only the text between
    the <TEXT> tags of the section identified as <TYPE>10-K, discarding attachments,
    base64 images, and secondary XML documents in the submission file.

    Args:
        raw_submission (str): The full text content of the submission (.txt).

    Returns:
        str: The HTML/text content of the 10-K report only.
    """
    # We use a regex to find the 10-K section and capture only the main text 
    match = re.search(r'<DOCUMENT>\s*<TYPE>10-K.*?(<TEXT>.*?</TEXT>)', raw_submission, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return raw_submission


def linearize_sec_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Transforms HTML tables into linearized text format readable by LLMs.

    Iterates over all <table> tags and converts rows (tr) and cells (td/th) into strings
    separated by the '|' character. Handles critical cases such as dollar signs
    separated by digits and negative numbers in parentheses. Adds the
    [START TABLE] and [END TABLE] markers to improve semantic retrieval (RAG).

    Args:
        soup (BeautifulSoup): BeautifulSoup object of the HTML document.

    Returns:
        BeautifulSoup: The modified soup object with the tables replaced by text.
    """
    # We iterate over all the tables in the document
    for table in soup.find_all("table"):
        linearized_rows = []
        # For each row in the table, we extract the cells and clean the text
        for tr in table.find_all("tr"):
            # We extract all cells, both <td> and <th>, so as not to lose headers or important data
            cells = tr.find_all(["td", "th"])
            row_values = []
            # We clean the text of each cell, removing extra spaces and newlines, and ignoring empty cells
            for c in cells:
                # We get the text, replace newlines with spaces, and strip leading/trailing whitespace
                text = c.get_text(separator=" ", strip=True).replace('\n', ' ')
                # We ignore cells that are empty or contain only non-breaking spaces or similar invisible characters
                if text and text not in ["", "\xa0", "​", "_"]:
                    row_values.append(text)
            
            # Now we have a list of cleaned cell values for the row. We need to handle special cases:
            merged_row = []
            i = 0
            # We iterate over the cell values and look for patterns that indicate a split of a single logical value 
            # into multiple cells, such as:
            while i < len(row_values):
                val = row_values[i]
                # If we find a dollar sign followed by a number, we merge them into a single value (e.g., "$" + "100" -> "$100")
                if val == '$' and i + 1 < len(row_values):
                    merged_row.append('$' + row_values[i+1])
                    i += 2
                # If we find an opening parenthesis followed by a number and a closing parenthesis, we merge them into a single value (e.g., "(" + "67" + ")" -> "(67)")
                elif val == '(' and i + 2 < len(row_values) and row_values[i+2] == ')':
                    merged_row.append(f"({row_values[i+1]})")
                    i += 3
                # In all other cases, we simply add the value to the merged row
                else:
                    merged_row.append(val)
                    i += 1
                    
            # After processing the row, if we have any values left, we join them with " | " and add them to the list of linearized rows
            if merged_row:
                linearized_rows.append(" | ".join(merged_row))
        
        # After processing all rows of the table, if we have any linearized rows, we create a new <div> tag to replace the table
        if linearized_rows:
            # We create a new <div> tag to replace the table, and we add the [START TABLE] and [END TABLE] markers to improve semantic retrieval (RAG)
            replacement = soup.new_tag("div")
            # We join the linearized rows with newlines to maintain the vertical structure of the table, and we add the markers around it
            table_text = "\n\n[START TABLE]\n" + "\n".join(linearized_rows) + "\n[END TABLE]\n\n"
            # We set the text of the replacement div to the linearized table text, and we replace the original table with the new div
            replacement.string = table_text
            # We replace the original table with the new div containing the linearized text
            table.replace_with(replacement)
        # If the table is empty or we couldn't extract any meaningful data, we simply remove it from the document
        else:
            table.decompose()
            
    return soup


def clean_sec_text(raw_html: str) -> str:
    """
    Runs the complete SEC text cleansing and structuring pipeline.

    Removes non-textual tags (script, style), invokes table linearization,
    eliminates residual HTML attributes and XBRL prefixes. Additionally, identifies and marks
    the 'Item' sections of the report to facilitate context-aware chunking.

    Args:
        raw_html (str): The raw HTML content of the 10-K.

    Returns:
        str: Cleaned, normalized, and structured text ready for ingestion.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    # Remove non-text tags (script, style, header, footer, ix:header)
    for element in soup(["script", "style", "header", "footer", "ix:header"]):
        element.decompose()
    # We linearize tables before extracting text, so as to preserve their structure and semantics.
    soup = linearize_sec_tables(soup)
    # After cleaning the document and linearizing the tables, we extract the text. We use a double newline as a separator to maintain some structure in the text, especially for tables and sections.
    text = soup.get_text(separator="\n\n")
    # We perform additional cleaning on the extracted text to remove any residual HTML attributes, XBRL prefixes, and to identify and mark the 'Item' sections of the report. 
    # We also normalize whitespace and replace non-breaking spaces with regular spaces.
    text = re.sub(r'[a-zA-Z\-]+="[^"]*"', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\b(us-gaap|srt|dei|iso4217):\S+', '', text)
    text = re.sub(r'\n\s*(Item\s+[1-9][A-Z]?[\.\s]+[A-Za-z].*?)\n', r'\n[SECTION: \1]\n', text, flags=re.IGNORECASE)
    # We also normalize multiple newlines and spaces to avoid excessive whitespace in the final text.
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', '  ', text) 
    text = text.replace('\xa0', ' ')
    
    return text.strip()