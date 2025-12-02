"""
Span Labeling Script for MTA Transit Codes
Creates silver data annotations with span positions for NER tasks.

This script extracts transit code spans from the Header column and creates
annotations in the format: [{"start": int, "end": int, "label": "TRANSIT_CODE"}]
"""

import pandas as pd
import re
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def load_bus_codes(filepath: str = "Data/bus_codes.csv") -> Set[str]:
    """Load valid bus codes from CSV file."""
    df = pd.read_csv(filepath)
    return set(df['route_short_name'].astype(str).tolist())


def load_subway_codes(filepath: str = "Data/subway_codes.csv") -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load valid subway codes from CSV file.
    
    Returns:
        Tuple of (letter_codes, digit_codes, special_codes)
        - letter_codes: Single letter codes like A, B, C
        - digit_codes: Single digit codes like 1, 2, 3
        - special_codes: Multi-character codes like SIR, 5X, 6X, 7X, FX
    """
    df = pd.read_csv(filepath, sep=';')
    all_codes = set(df['route_short_name'].astype(str).tolist())
    
    letter_codes = set()
    digit_codes = set()
    special_codes = set()
    
    for code in all_codes:
        if len(code) == 1 and code.isalpha():
            letter_codes.add(code)
        elif len(code) == 1 and code.isdigit():
            digit_codes.add(code)
        else:
            # Multi-character codes: SIR, 5X, 6X, 7X, FX, S (shuttle)
            special_codes.add(code)
    
    return letter_codes, digit_codes, special_codes


def parse_affected_column(affected_str: str) -> List[str]:
    """Parse the Affected column JSON string to get list of codes."""
    try:
        if pd.isna(affected_str):
            return []
        return json.loads(affected_str)
    except (json.JSONDecodeError, TypeError):
        return []


def normalize_bus_code(code: str) -> str:
    """
    Normalize bus code for matching against Affected column.
    Handles variations like Q44-SBS -> Q44, Q8 -> Q08
    """
    # Remove -SBS suffix for base code
    base_code = code.replace('-SBS', '')
    return base_code


def extract_bus_code_spans(text: str, valid_bus_codes: Set[str], affected_codes: List[str]) -> List[Dict]:
    """
    Extract bus code spans from text.
    Bus codes are unambiguous multi-character codes like Q65, B44-SBS, BxM10, etc.
    """
    spans = []
    
    # Create normalized set of valid bus codes for matching
    valid_bus_codes_upper = {code.upper() for code in valid_bus_codes}
    affected_codes_upper = {code.upper() for code in affected_codes}
    
    # Pattern for bus codes - they start with specific prefixes
    # Covers: B, Bx, BxM, M, Q, QM, S, SIM, X, BM prefixes
    # Also handles -SBS suffix and special codes like Bx4A, M14A-SBS, Bx18A, Bx18B, Q20A, Q20B
    # Note: Made case-insensitive to match both Bx and BX
    bus_pattern = r'\b(B(?:x(?:M)?)?|M|Q(?:M)?|S(?:IM)?|X|BM)\d+[A-Z]?(?:-SBS)?\b'
    
    for match in re.finditer(bus_pattern, text, re.IGNORECASE):
        code = match.group(0)
        code_upper = code.upper()
        base_code = normalize_bus_code(code).upper()
        # Verify it's a valid bus code (check both exact match and base code without -SBS)
        # Also check if it matches any affected code (normalized)
        if code_upper in valid_bus_codes_upper or base_code in valid_bus_codes_upper or base_code in affected_codes_upper:
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "label": "TRANSIT_CODE",
                "text": code
            })
    
    # Also check for special bus codes that don't match the main pattern
    special_bus_patterns = [
        r'\bBEDLOOP\b',
        r'\bBJ-MW\b',
        r'\bMW-ML\b',
        r'\bF1\b',
        r'\bD90\b', r'\bD99\b',
        r'\bJ90\b', r'\bJ99\b',
        r'\bL90\b', r'\bL91\b', r'\bL92\b',
    ]
    
    for pattern in special_bus_patterns:
        for match in re.finditer(pattern, text):
            code = match.group(0)
            if code in valid_bus_codes:
                # Check if this span is not already captured
                if not any(s['start'] == match.start() and s['end'] == match.end() for s in spans):
                    spans.append({
                        "start": match.start(),
                        "end": match.end(),
                        "label": "TRANSIT_CODE",
                        "text": code
                    })
    
    return spans


def extract_letter_subway_spans(text: str, letter_codes: Set[str], affected_codes: List[str]) -> List[Dict]:
    """
    Extract single-letter subway code spans from text.
    Only matches capital letters that are valid subway codes.
    Filters out false positives like "E 149 St" (East street).
    Only extracts letters that are in the affected codes OR clearly in transit context.
    """
    spans = []
    
    # Get letter codes that are in the affected list for this row
    affected_letters = set(code for code in affected_codes if code in letter_codes)
    
    # Pattern for single capital letters that are word boundaries
    # Valid letters: A, B, C, D, E, F, G, H (Rockaway Park Shuttle), J, L, M, N, Q, R, W, Z
    letter_pattern = r'\b([A-GHJLMNQRWZ])\b'
    
    for match in re.finditer(letter_pattern, text):
        letter = match.group(1)
        if letter in letter_codes:
            # Check for false positives - E/W/N/S followed by street numbers
            # Like "E 149 St" (East), "W 42 St" (West), etc.
            after_text = text[match.end():match.end()+20]
            before_text = text[max(0, match.start()-20):match.start()]
            
            # Skip if followed by a number and then St/Av/Ave/Street/Avenue
            if letter in {'E', 'W', 'N', 'S'} and re.match(r'\s+\d+\s*(St|Av|Ave|Street|Avenue)\b', after_text, re.IGNORECASE):
                # Only skip if this letter is NOT in the affected codes for this row
                if letter not in affected_letters:
                    continue
            
            # If letter is in affected codes, always extract it
            if letter in affected_letters:
                spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": "TRANSIT_CODE",
                    "text": letter
                })
            else:
                # For letters NOT in affected, only extract if in clear transit context
                # Context: followed by train/trains/line/service or preceded by similar
                in_transit_context = (
                    re.match(r'\\s*(train|trains|line|service)\\b', after_text, re.IGNORECASE) or
                    re.search(r'(train|trains|line|service)\\s*$', before_text, re.IGNORECASE) or
                    re.search(r'bound\\s*$', before_text, re.IGNORECASE)
                )
                # Don't add if not in transit context - this prevents false positives
                # like extracting R from "R line" when R is mentioned but not affected
                # We'll skip these - they're mentioned but not the focus of the alert
    
    return spans


def extract_special_subway_spans(text: str, special_codes: Set[str], affected_codes: List[str]) -> List[Dict]:
    """
    Extract special multi-character subway code spans.
    Includes: SIR, 5X, 6X, 7X, FX, S (shuttle), FS (Franklin Av Shuttle), H (Rockaway Park Shuttle)
    """
    spans = []
    
    # Pattern for special codes (excluding SIR which needs special handling)
    special_pattern = r'\b([567]X|FX|FS)\b'
    
    for match in re.finditer(special_pattern, text):
        code = match.group(0)
        if code in special_codes:
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "label": "TRANSIT_CODE",
                "text": code
            })
    
    # Handle SIR (Staten Island Railway) - extract as SI (official code in Affected column)
    sir_pattern = r'\b(SIR)\b'
    for match in re.finditer(sir_pattern, text, re.IGNORECASE):
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "label": "TRANSIT_CODE",
            "text": "SI"  # Normalized to SI code (official code in Affected)
        })
    
    # Handle "Franklin Av Shuttle" pattern - extract as FS
    franklin_pattern = r'\b(Franklin\s+Av(?:enue)?\s+Shuttle)\b'
    for match in re.finditer(franklin_pattern, text, re.IGNORECASE):
        # We mark the span of the whole phrase but label it as the shuttle code
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "label": "TRANSIT_CODE",
            "text": "FS"  # Normalized to FS code
        })
    
    # Handle "Rockaway Park Shuttle" pattern - extract as H
    # H is the internal route designator for Rockaway Park Shuttle
    rockaway_pattern = r'\b(Rockaway\s+Park\s+Shuttle)\b'
    for match in re.finditer(rockaway_pattern, text, re.IGNORECASE):
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "label": "TRANSIT_CODE",
            "text": "H"  # Normalized to H code
        })
    
    # Handle "42 St Shuttle" pattern - extract as GS
    # GS is the route code for the 42nd Street Shuttle
    shuttle_42_pattern = r'\b(42\s*St\s+Shuttle)\b'
    for match in re.finditer(shuttle_42_pattern, text, re.IGNORECASE):
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "label": "TRANSIT_CODE",
            "text": "GS"  # Normalized to GS code
        })
    
    # Handle shuttle 'S' - appears as "S train" or "S shuttle"
    # This is tricky because S is also a common letter
    shuttle_pattern = r'\b(S)\s+(?:train|trains|shuttle)\b'
    for match in re.finditer(shuttle_pattern, text, re.IGNORECASE):
        spans.append({
            "start": match.start(1),
            "end": match.end(1),
            "label": "TRANSIT_CODE",
            "text": "S"
        })
    
    return spans


def extract_digit_subway_spans(text: str, digit_codes: Set[str], affected_codes: List[str]) -> List[Dict]:
    """
    Extract single-digit subway code spans from text with context-aware rules.
    
    Rules:
    1. Match if followed by: track, train, trains, service, line
    2. Match if preceded/followed by *bound pattern (Northbound, Bronx-bound, etc.)
    3. Match if at sentence start
    4. Fallback: If digit is in Affected but not yet matched, find and label it
    """
    spans = []
    matched_digits = set()
    
    # Get affected digit codes
    affected_digits = set(code for code in affected_codes if code in digit_codes)
    
    # Rule 1: Digit followed by transit keywords
    # Pattern matches: "2 trains", "3 train", "5 service", "6 line"
    # NOTE: "track" is excluded here as it's ambiguous - "Track 2" often means platform track, not line 2
    # We'll handle track separately only if the digit is in Affected
    keyword_after_pattern = r'\b([1-7])\s+(?:train|trains|service|line)\b'
    for match in re.finditer(keyword_after_pattern, text, re.IGNORECASE):
        digit = match.group(1)
        if digit in digit_codes:
            spans.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "TRANSIT_CODE",
                "text": digit
            })
            matched_digits.add(digit)
    
    # Rule 1b: Digit preceded by transit keywords (excluding track)
    # Pattern matches: "train 2", "trains 2 3 4"
    keyword_before_pattern = r'(?:train|trains|service|line)\s+([1-7])\b'
    for match in re.finditer(keyword_before_pattern, text, re.IGNORECASE):
        digit = match.group(1)
        if digit in digit_codes:
            # Check if this exact span is already captured
            if not any(s['start'] == match.start(1) and s['end'] == match.end(1) for s in spans):
                spans.append({
                    "start": match.start(1),
                    "end": match.end(1),
                    "label": "TRANSIT_CODE",
                    "text": digit
                })
                matched_digits.add(digit)
    
    # Rule 2: Digit near *bound patterns
    # Patterns: "Northbound 2", "Manhattan-bound 3", "2 trains Bronx-bound"
    bound_patterns = [
        # Direction word followed by digit
        r'(?:Northbound|Southbound|Eastbound|Westbound|Uptown|Downtown)\s+([1-7])\b',
        # Place-bound followed by digit (handles "Jamaica-bound 7")
        r'\b[\w-]+-bound\s+([1-7])\b',
        # Digit followed by bound pattern (handles "2 3 trains are delayed ... bound")
        r'\b([1-7])\s+(?=[\w\s]*(?:Northbound|Southbound|Eastbound|Westbound|bound))',
    ]
    
    for pattern in bound_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            digit = match.group(1)
            if digit in digit_codes:
                if not any(s['start'] == match.start(1) and s['end'] == match.end(1) for s in spans):
                    spans.append({
                        "start": match.start(1),
                        "end": match.end(1),
                        "label": "TRANSIT_CODE",
                        "text": digit
                    })
                    matched_digits.add(digit)
    
    # Rule 3: Digit at sentence start
    # Pattern: Start of string or after sentence-ending punctuation
    sentence_start_pattern = r'(?:^|[.!?]\s+)([1-7])\b'
    for match in re.finditer(sentence_start_pattern, text):
        digit = match.group(1)
        if digit in digit_codes:
            if not any(s['start'] == match.start(1) and s['end'] == match.end(1) for s in spans):
                spans.append({
                    "start": match.start(1),
                    "end": match.end(1),
                    "label": "TRANSIT_CODE",
                    "text": digit
                })
                matched_digits.add(digit)
    
    # Rule 4: Fallback - if affected digit not matched, find any occurrence
    # This handles cases where multiple digits appear together like "2 3 4 5"
    unmatched_digits = affected_digits - matched_digits
    
    if unmatched_digits:
        # Find all digit occurrences in text
        for digit in unmatched_digits:
            digit_pattern = rf'\b{digit}\b'
            for match in re.finditer(digit_pattern, text):
                # Check if this exact span is already captured
                if not any(s['start'] == match.start() and s['end'] == match.end() for s in spans):
                    # Additional check: not part of a street name like "96 St" or time like "10:30"
                    # Look at surrounding context
                    before = text[max(0, match.start()-3):match.start()]
                    after = text[match.end():min(len(text), match.end()+5)]
                    
                    # Skip if looks like a street number (followed by St, Av, Ave, Street, Avenue)
                    if re.search(r'^\s*(St|Av|Ave|Street|Avenue)\b', after, re.IGNORECASE):
                        continue
                    # Skip if preceded by a number (part of multi-digit number like "96")
                    if re.search(r'\d$', before):
                        continue
                    # Skip if followed by a digit (part of multi-digit number)
                    if re.search(r'^\d', after):
                        continue
                    
                    spans.append({
                        "start": match.start(),
                        "end": match.end(),
                        "label": "TRANSIT_CODE",
                        "text": digit
                    })
    
    return spans


def extract_slash_separated_codes(text: str, all_codes: Set[str]) -> List[Dict]:
    """
    Extract codes separated by slashes like "2/3 trains", "B/Q trains".
    Each code gets its own span.
    """
    spans = []
    
    # Pattern for slash-separated codes followed by transit keywords
    slash_pattern = r'\b([A-Z1-7](?:/[A-Z1-7])+)\s+(?:train|trains|bus|buses|line|service)\b'
    
    for match in re.finditer(slash_pattern, text, re.IGNORECASE):
        codes_str = match.group(1)
        codes = codes_str.split('/')
        
        # Calculate position of each code within the slash group
        current_pos = match.start(1)
        for code in codes:
            if code in all_codes:
                spans.append({
                    "start": current_pos,
                    "end": current_pos + len(code),
                    "label": "TRANSIT_CODE",
                    "text": code
                })
            current_pos += len(code) + 1  # +1 for the slash
    
    return spans


def extract_all_spans(text: str, agency: str, affected_codes: List[str],
                      valid_bus_codes: Set[str], letter_codes: Set[str],
                      digit_codes: Set[str], special_codes: Set[str]) -> List[Dict]:
    """
    Extract all transit code spans from text based on agency type.
    """
    spans = []
    
    # Create set of all valid codes for slash pattern
    all_codes = valid_bus_codes | letter_codes | digit_codes | special_codes
    
    if agency == "NYCT Bus":
        # For buses, use straightforward pattern matching
        spans.extend(extract_bus_code_spans(text, valid_bus_codes, affected_codes))
    
    elif agency == "NYCT Subway":
        # For subways, use context-aware rules
        spans.extend(extract_letter_subway_spans(text, letter_codes, affected_codes))
        spans.extend(extract_special_subway_spans(text, special_codes, affected_codes))
        spans.extend(extract_digit_subway_spans(text, digit_codes, affected_codes))
    
    # Handle slash-separated codes for both agencies
    slash_spans = extract_slash_separated_codes(text, all_codes)
    for span in slash_spans:
        # Only add if not already captured
        if not any(s['start'] == span['start'] and s['end'] == span['end'] for s in spans):
            spans.append(span)
    
    # Remove duplicates and sort by start position
    unique_spans = []
    seen = set()
    for span in sorted(spans, key=lambda x: x['start']):
        key = (span['start'], span['end'])
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)
    
    return unique_spans


def normalize_code_for_comparison(code: str) -> str:
    """
    Normalize code for comparison between extracted and affected.
    Handles variations like:
    - Q44-SBS <-> Q44
    - Q08 <-> Q8 (leading zeros)
    - BX19 <-> Bx19 (case differences)
    """
    # Remove -SBS suffix
    normalized = code.replace('-SBS', '')
    # Uppercase for case-insensitive comparison
    normalized = normalized.upper()
    # Remove leading zeros from numbers in codes (Q08 -> Q8)
    # Match pattern like Q08, B01, etc.
    match = re.match(r'^([A-Za-z]+)0*(\d+)([A-Za-z]?)$', normalized)
    if match:
        prefix, num, suffix = match.groups()
        normalized = f"{prefix}{int(num)}{suffix}"
    return normalized


def validate_spans(spans: List[Dict], affected_codes: List[str]) -> Dict:
    """
    Validate extracted spans against affected codes.
    Handles code normalization for variations like Q44-SBS vs Q44.
    
    Returns:
        Dictionary with validation results
    """
    # Normalize extracted codes
    extracted_normalized = {}  # normalized -> original
    for span in spans:
        original = span['text']
        normalized = normalize_code_for_comparison(original)
        extracted_normalized[normalized] = original
    
    # Normalize affected codes
    affected_normalized = {}  # normalized -> original
    for code in affected_codes:
        normalized = normalize_code_for_comparison(code)
        affected_normalized[normalized] = code
    
    extracted_set = set(extracted_normalized.keys())
    affected_set = set(affected_normalized.keys())
    
    # Codes in Affected but not extracted (using normalized comparison)
    missing_normalized = affected_set - extracted_set
    missing_codes = [affected_normalized[n] for n in missing_normalized]
    
    # Codes extracted but not in Affected
    extra_normalized = extracted_set - affected_set
    extra_codes = [extracted_normalized[n] for n in extra_normalized]
    
    # Correctly matched codes
    matched_normalized = affected_set & extracted_set
    matched_codes = [affected_normalized[n] for n in matched_normalized]
    
    return {
        "matched": list(matched_codes),
        "missing": list(missing_codes),
        "extra": list(extra_codes),
        "is_complete_match": len(missing_codes) == 0 and len(extra_codes) == 0,
        "coverage": len(matched_codes) / len(affected_set) if affected_set else 1.0
    }


def process_dataset(input_path: str, output_path: str, mismatch_path: str):
    """
    Process the entire dataset and create span annotations.
    Updates Affected column based on extracted codes:
    - Removes rows with no extracted route codes
    - Removes missing codes from Affected (codes not found in Header)
    - Adds extra codes to Affected (codes found in Header but not in original Affected)
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Loading transit code files...")
    valid_bus_codes = load_bus_codes()
    letter_codes, digit_codes, special_codes = load_subway_codes()
    
    print(f"Loaded {len(valid_bus_codes)} bus codes")
    print(f"Loaded {len(letter_codes)} letter subway codes: {letter_codes}")
    print(f"Loaded {len(digit_codes)} digit subway codes: {digit_codes}")
    print(f"Loaded {len(special_codes)} special subway codes: {special_codes}")
    
    # Process each row - first pass to extract spans
    span_annotations = []
    updated_affected = []
    rows_to_keep = []
    modifications = []  # Track all modifications for reporting
    
    total_rows = len(df)
    print(f"\nProcessing {total_rows} rows...")
    
    rows_removed_no_codes = 0
    rows_affected_updated = 0
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0: # type: ignore
            print(f"Progress: {idx}/{total_rows} ({100*idx/total_rows:.1f}%)") # type: ignore
        
        header = str(row['header']) if pd.notna(row['header']) else ""
        agency = row['agency']
        affected_codes = parse_affected_column(row['affected'])
        
        # Extract spans
        spans = extract_all_spans(
            header, agency, affected_codes,
            valid_bus_codes, letter_codes, digit_codes, special_codes
        )
        
        # Get extracted codes in order of occurrence (by start position), keeping unique values
        seen = set()
        extracted_codes = []
        for s in sorted(spans, key=lambda x: x['start']):
            if s['text'] not in seen:
                seen.add(s['text'])
                extracted_codes.append(s['text'])
        
        # Step 1: Remove rows with no extracted route codes
        if len(extracted_codes) == 0:
            rows_removed_no_codes += 1
            modifications.append({
                "index": idx,
                "alert_id": row.get('alert_id', ''),
                "agency": agency,
                "header": header,
                "original_affected": row['affected'],
                "new_affected": "[]",
                "extracted_codes": [],
                "missing_codes": affected_codes,
                "extra_codes": [],
                "action": "REMOVED - No codes extracted"
            })
            continue
        
        # Mark row to keep
        rows_to_keep.append(idx)
        
        # Step 2 & 3: Update Affected column based on extracted codes
        # The new Affected should be exactly what was extracted
        # This removes missing codes and adds extra codes automatically
        original_affected_normalized = set(normalize_code_for_comparison(c) for c in affected_codes)
        new_affected_normalized = set(normalize_code_for_comparison(c) for c in extracted_codes)
        
        # Calculate what was removed and added
        missing_normalized = original_affected_normalized - new_affected_normalized
        extra_normalized = new_affected_normalized - original_affected_normalized
        
        # Get original codes for missing/extra
        missing_codes = [c for c in affected_codes if normalize_code_for_comparison(c) in missing_normalized]
        extra_codes = [c for c in extracted_codes if normalize_code_for_comparison(c) in extra_normalized]
        
        if missing_codes or extra_codes:
            rows_affected_updated += 1
            modifications.append({
                "index": idx,
                "alert_id": row.get('alert_id', ''),
                "agency": agency,
                "header": header,
                "original_affected": row['affected'],
                "new_affected": json.dumps(extracted_codes),
                "extracted_codes": extracted_codes,
                "missing_codes": missing_codes,
                "extra_codes": extra_codes,
                "action": "UPDATED"
            })
        
        # Format output with type and value fields
        output_spans = [{"start": s['start'], "end": s['end'], "type": "ROUTE", "value": s['text']} for s in spans]
        span_annotations.append(json.dumps(output_spans))
        
        # Store updated Affected as JSON list
        updated_affected.append(json.dumps(extracted_codes))
    
    # Filter dataframe to keep only rows with extracted codes
    df = df.loc[rows_to_keep].copy()
    df = df.reset_index(drop=True)
    
    # Add/update columns
    df['affected'] = updated_affected
    df['affected_spans'] = span_annotations
    
    # Reorder columns to place affected_spans right after affected
    cols = df.columns.tolist()
    if 'affected' in cols and 'affected_spans' in cols:
        cols.remove('affected_spans')
        affected_idx = cols.index('affected')
        cols.insert(affected_idx + 1, 'affected_spans')
        df = df[cols]
    
    # Calculate statistics
    final_rows = len(df)
    
    print(f"\n{'='*60}")
    print("PROCESSING STATISTICS")
    print(f"{'='*60}")
    print(f"Total rows in input: {total_rows}")
    print(f"Rows removed (no extracted codes): {rows_removed_no_codes} ({100*rows_removed_no_codes/total_rows:.2f}%)")
    print(f"Rows with updated Affected column: {rows_affected_updated} ({100*rows_affected_updated/final_rows:.2f}%)")
    print(f"Final rows in output: {final_rows}")
    
    # Save main output
    print(f"\nSaving annotated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Save modifications report
    if modifications:
        print(f"Saving modifications report to {mismatch_path}...")
        mismatch_df = pd.DataFrame(modifications)
        mismatch_df.to_csv(mismatch_path, index=False)
        
        # Print sample modifications
        print(f"\n{'='*60}")
        print(f"SAMPLE MODIFICATIONS (first 10 of {len(modifications)})")
        print(f"{'='*60}")
        for m in modifications[:10]:
            print(f"\nIndex: {m['index']} | Action: {m['action']}")
            print(f"Agency: {m['agency']}")
            print(f"Header: {m['header'][:100]}...")
            print(f"Original Affected: {m['original_affected']}")
            print(f"New Affected: {m['new_affected']}")
            print(f"Missing (removed): {m['missing_codes']}")
            print(f"Extra (added): {m['extra_codes']}")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    return df, modifications


if __name__ == "__main__":
    input_path = "Preprocessed/MTA_Data_preprocessed.csv"
    output_path = "Preprocessed/MTA_Data_preprocessed_routespans.csv"
    mismatch_path = "Preprocessed/span_labeling_mismatches.csv"
    
    df, modifications = process_dataset(input_path, output_path, mismatch_path)
