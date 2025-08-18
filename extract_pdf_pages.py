#!/usr/bin/env python3
"""
Script to extract pages 23 and below from a PDF file and create a new PDF.
"""

import sys
from pypdf import PdfReader, PdfWriter
import os

def extract_pages_from_pdf(input_file, output_file, start_page=23):
    """
    Extract pages from start_page onwards from input_file and save to output_file.
    
    Args:
        input_file (str): Path to the input PDF file
        output_file (str): Path to the output PDF file
        start_page (int): Starting page number (1-based indexing)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False
            
        # Create a PDF reader object
        reader = PdfReader(input_file)
        
        # Get total number of pages
        total_pages = len(reader.pages)
        print(f"Total pages in PDF: {total_pages}")
        
        # Convert to 0-based indexing
        start_index = start_page - 1
        
        # Check if start page is valid
        if start_index >= total_pages:
            print(f"Error: Start page {start_page} is beyond the total pages ({total_pages}).")
            return False
        
        # Create a PDF writer object
        writer = PdfWriter()
        
        # Add pages from start_page onwards
        pages_added = 0
        for page_num in range(start_index, total_pages):
            page = reader.pages[page_num]
            writer.add_page(page)
            pages_added += 1
        
        # Write the new PDF
        with open(output_file, 'wb') as output_pdf:
            writer.write(output_pdf)
        
        print(f"Successfully extracted {pages_added} pages (from page {start_page} onwards)")
        print(f"New PDF saved as: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

def main():
    input_file = "npre.2008.1244.2.pdf"
    output_file = "npre.2008.1244.2_pages_23_onwards.pdf"
    start_page = 23
    
    print(f"Extracting pages {start_page} and below from '{input_file}'...")
    success = extract_pages_from_pdf(input_file, output_file, start_page)
    
    if success:
        print("PDF extraction completed successfully!")
    else:
        print("PDF extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 