# RNA MSA Processing Protocol: Chunking and Filtering for Long RNA Sequences

## Background and Objectives

We need to process a large collection of Multiple Sequence Alignments (MSAs) representing very long RNAs (3,800-4,300 nucleotides) to make them compatible with our computational pipeline, which has a 750 nucleotide limit for test sequences. The goal is to reduce our ~60,000 MSAs to ~1,000-2,000 most informative alignments while preserving biological diversity and functional information.

## Recommended Two-Step Approach

1. **Chunking**: Split long RNA sequences into overlapping segments of manageable size
2. **Filtering**: Remove redundant sequences while preserving biological diversity

This document outlines the implementation details for both strategies.

## Step 1: Chunking Long RNA Sequences

### Parameters and Justification

- **Chunk size: 600 nucleotides** - Balances computational efficiency with capturing structural elements
- **Overlap: 150 nucleotides** - Ensures structural motifs near chunk boundaries aren't lost
- **Processing: All sequences > 750 nt** - Apply chunking to any sequence exceeding our pipeline limit

### Implementation

```python
#!/usr/bin/env python3
"""
RNA Sequence Chunker - Splits long RNA sequences into overlapping chunks
Usage: python chunk_rna.py --input long_rnas.fasta --output chunked.fasta --size 600 --overlap 150
"""

import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def chunk_sequence(record, chunk_size, overlap):
    """Split a sequence into overlapping chunks."""
    seq_length = len(record.seq)
    chunks = []
    
    for start in range(0, seq_length - overlap, chunk_size - overlap):
        end = min(start + chunk_size, seq_length)
        chunk_seq = record.seq[start:end]
        
        # Only keep chunks of sufficient length (at least 75% of target size)
        if len(chunk_seq) >= chunk_size * 0.75:
            chunk_id = f"{record.id}_chunk_{start+1}-{end}"
            chunk_desc = f"Chunk {start+1}-{end} from {record.description}"
            chunk_record = SeqRecord(chunk_seq, id=chunk_id, description=chunk_desc)
            chunks.append(chunk_record)
        
        # Stop if we've reached the end
        if end == seq_length:
            break
            
    return chunks

def main():
    parser = argparse.ArgumentParser(description='Split RNA sequences into overlapping chunks')
    parser.add_argument('--input', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output FASTA file')
    parser.add_argument('--size', type=int, default=600, help='Chunk size (default: 600)')
    parser.add_argument('--overlap', type=int, default=150, help='Overlap between chunks (default: 150)')
    parser.add_argument('--min_length', type=int, default=750, 
                       help='Minimum sequence length to apply chunking (default: 750)')
    args = parser.parse_args()
    
    chunks = []
    with open(args.input) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Only chunk sequences longer than min_length
            if len(record.seq) > args.min_length:
                seq_chunks = chunk_sequence(record, args.size, args.overlap)
                chunks.extend(seq_chunks)
            else:
                # Keep short sequences as-is
                chunks.append(record)
    
    # Write all chunks to output file
    with open(args.output, "w") as output_handle:
        SeqIO.write(chunks, output_handle, "fasta")
    
    print(f"Processed {len(chunks)} chunks from input sequences")

if __name__ == "__main__":
    main()
```

### Batch Processing Script

For processing multiple files:

```bash
#!/bin/bash
# Batch process all RNA FASTA files in a directory

CHUNK_SIZE=600
OVERLAP=150
MIN_LENGTH=750

mkdir -p chunked

for file in *.fasta; do
  echo "Processing $file..."
  python chunk_rna.py --input "$file" \
                      --output "chunked/${file%.fasta}_chunked.fasta" \
                      --size $CHUNK_SIZE \
                      --overlap $OVERLAP \
                      --min_length $MIN_LENGTH
done

# Combine all chunked files (if needed)
cat chunked/*_chunked.fasta > all_chunked.fasta
```

## Step 2: Filtering MSAs

After chunking, we need to filter the MSAs to:
1. Remove sequences with poor coverage
2. Cluster similar sequences to reduce redundancy
3. Select representative MSAs based on information content

### 2.1: Coverage-Based Filtering

**Parameters:**
- **Minimum coverage: 70%** - Ensures sequences cover a substantial portion of the query
- **Tool: usearch** - Efficient for large datasets

```bash
# Filter sequences with insufficient coverage of the query
usearch -search_pcr query.fasta -db chunked_msas.fasta -strand both -maxdiffs 100 -minamp 0.7 -ampout filtered_by_coverage.fasta
```

### 2.2: Identity-Based Clustering

**Parameters:**
- **Identity threshold: 80%** - Balances diversity preservation with redundancy removal
- **Tool: cd-hit-est** - Optimized for nucleotide sequences

```bash
# Cluster sequences by identity
cd-hit-est -i filtered_by_coverage.fasta \
           -o clustered.fasta \
           -c 0.80 \
           -n 8 \
           -d 0 \
           -M 16000 \
           -T 8 \
           -g 1

# Extract cluster representatives
grep ">" clustered.fasta | cut -d ">" -f 2 > representative_sequences.txt
```

### 2.3: Information Content Filtering

To select the most informative MSAs, rank them by information content:

```python
#!/usr/bin/env python3
"""
MSA Information Content Calculator
Ranks MSAs by information content to identify the most informative alignments
"""

import argparse
import numpy as np
from Bio import AlignIO
from multiprocessing import Pool

def shannon_entropy(column):
    """Calculate Shannon entropy for an alignment column."""
    bases = 'ACGTU-'
    base_frequencies = [column.count(base)/len(column) for base in bases]
    # Remove zero frequencies (0 * log(0) = 0)
    base_frequencies = [f for f in base_frequencies if f > 0]
    return -sum(f * np.log2(f) for f in base_frequencies)

def gap_fraction(column):
    """Calculate fraction of gaps in a column."""
    return column.count('-') / len(column)

def calculate_msa_score(alignment):
    """Calculate information content score for an MSA."""
    # Get alignment columns
    alignment_length = alignment.get_alignment_length()
    columns = [alignment[:, i] for i in range(alignment_length)]
    
    # Calculate entropy for each column
    entropies = [shannon_entropy(col) for col in columns]
    
    # Calculate gap fractions
    gap_fractions = [gap_fraction(col) for col in columns]
    
    # Calculate normalized information content score
    # Higher score = more informative alignment
    # We penalize gaps and reward columns with intermediate entropy (indicating conservation patterns)
    scores = [(1 - gap_fractions[i]) * (1 - abs(entropies[i]/2 - 1)) for i in range(alignment_length)]
    
    # Overall alignment score
    avg_score = np.mean(scores)
    seq_diversity = len(set(str(record.seq) for record in alignment))
    
    # Final score combines average column score and sequence diversity
    return {
        'id': alignment[0].id.split('_chunk_')[0] if '_chunk_' in alignment[0].id else alignment[0].id,
        'length': alignment_length,
        'num_sequences': len(alignment),
        'unique_sequences': seq_diversity,
        'avg_information': avg_score,
        'final_score': avg_score * np.log(seq_diversity + 1)
    }

def process_alignment(filename):
    """Process a single MSA file."""
    try:
        alignment = AlignIO.read(filename, "fasta")
        return calculate_msa_score(alignment)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate information content for MSAs')
    parser.add_argument('--input_dir', required=True, help='Directory containing MSA files')
    parser.add_argument('--output', required=True, help='Output TSV file with rankings')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads')
    args = parser.parse_args()
    
    import glob
    import os
    
    # Find all MSA files
    msa_files = glob.glob(os.path.join(args.input_dir, "*.fasta"))
    print(f"Found {len(msa_files)} MSA files")
    
    # Process MSAs in parallel
    with Pool(args.threads) as pool:
        results = pool.map(process_alignment, msa_files)
    
    # Filter out None results (errors)
    results = [r for r in results if r is not None]
    
    # Sort by final score (descending)
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Write results to TSV
    with open(args.output, 'w') as f:
        # Write header
        headers = list(results[0].keys())
        f.write('\t'.join(headers) + '\n')
        
        # Write data
        for result in results:
            f.write('\t'.join(str(result[h]) for h in headers) + '\n')
    
    print(f"Ranked {len(results)} MSAs by information content")

if __name__ == "__main__":
    main()
```

### 2.4: Final Selection

Select the top-ranked MSAs to reach your target count:

```bash
# Sort MSAs by score and select top 2000
sort -k5,5nr ranked_msas.tsv | head -n 2000 > top_msas.tsv

# Extract these MSAs
python extract_top_msas.py --rankings top_msas.tsv --msa_dir ./msas --output final_msas/
```

## Complete Pipeline Workflow

Here's the complete workflow combining chunking and filtering:

```bash
#!/bin/bash
# Full MSA processing pipeline: chunking and filtering

# Step 1: Chunk all sequences
echo "Chunking sequences..."
python chunk_rna.py --input all_rnas.fasta --output all_chunked.fasta --size 600 --overlap 150

# Step 2: Filter by coverage
echo "Filtering by coverage..."
usearch -search_pcr query.fasta -db all_chunked.fasta -strand both -maxdiffs 100 -minamp 0.7 -ampout coverage_filtered.fasta

# Step 3: Cluster by identity
echo "Clustering by identity..."
cd-hit-est -i coverage_filtered.fasta -o clustered.fasta -c 0.80 -n 8 -d 0 -M 16000 -T 8 -g 1

# Step 4: Split into individual MSAs (if needed)
echo "Splitting into individual MSAs..."
mkdir -p msas
python split_into_msas.py --input clustered.fasta --output_dir msas/

# Step 5: Calculate information content
echo "Calculating information content..."
python calculate_msa_info.py --input_dir msas/ --output ranked_msas.tsv --threads 16

# Step 6: Select top MSAs
echo "Selecting top MSAs..."
sort -k5,5nr ranked_msas.tsv | head -n 2000 > top_msas.tsv
mkdir -p final_msas
python extract_top_msas.py --rankings top_msas.tsv --msa_dir msas/ --output final_msas/

echo "Pipeline complete! Final MSAs are in final_msas/"
```

## Parameter Optimization Considerations

- **Chunk size (600 nt)**: Optimal for capturing local RNA structure while staying under 750 nt limit
- **Overlap (150 nt)**: ~25% of chunk size ensures structural elements aren't missed at boundaries
- **Coverage threshold (70%)**: Removes highly fragmented alignments without losing partial matches
- **Identity threshold (80%)**: Sweet spot between preserving diversity and removing redundancy

## Expected Outcomes

Starting with ~60,000 MSAs of 3,800-4,300 nt sequences:
1. Chunking will increase the count (~6-7 chunks per long RNA)
2. Coverage filtering will reduce by ~30-40%
3. Identity clustering will further reduce by ~40-50%
4. Information content filtering will select final 1,000-2,000 most informative MSAs

The final dataset will retain >90% of the functional diversity while being computationally tractable.

## Benchmarking and Validation

Test this pipeline on a smaller dataset first:

```bash
# Extract 100 random MSAs
grep "^>" all_rnas.fasta | shuf -n 100 | sed 's/>//' > sample.ids
python extract_sequences.py --input all_rnas.fasta --ids sample.ids --output sample.fasta

# Run pipeline on sample
./process_msas.sh sample.fasta

# Validate by comparing original vs processed data
python validate_filtering.py --original sample.fasta --processed final_msas/
```

This approach ensures we preserve the most informative regions of very long RNAs while making the data compatible with our computational pipeline's 750 nucleotide limit.