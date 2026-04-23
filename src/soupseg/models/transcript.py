"""
Transcript data model for SoupSeg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class Transcript:
    """
    Represents a single transcript with spatial coordinates and gene identity.
    
    Attributes:
        transcript_id: Unique identifier
        x: x coordinate (in pixels, Stereo-seq v1.3 = 0.5 um/pixel)
        y: y coordinate
        gene: Gene name
        count: UMI count (usually 1 for Stereo-seq)
        cell_id: Assigned cell ID (None if unassigned)
    """
    
    transcript_id: str
    x: float
    y: float
    gene: str
    count: int = 1
    cell_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transcript_id": self.transcript_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "gene": self.gene,
            "count": self.count,
            "cell_id": self.cell_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transcript:
        """Create from dictionary."""
        return cls(
            transcript_id=data["transcript_id"],
            x=data["x"],
            y=data["y"],
            gene=data["gene"],
            count=data.get("count", 1),
            cell_id=data.get("cell_id"),
        )


class TranscriptCollection:
    """
    A collection of Transcript objects with utility methods.
    """
    
    def __init__(self, transcripts: Optional[List[Transcript]] = None):
        self.transcripts: List[Transcript] = transcripts or []
    
    def add_transcript(self, transcript: Transcript):
        """Add a transcript."""
        self.transcripts.append(transcript)
    
    def __len__(self) -> int:
        return len(self.transcripts)
    
    def __iter__(self):
        return iter(self.transcripts)
    
    def __getitem__(self, idx: int) -> Transcript:
        return self.transcripts[idx]
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> TranscriptCollection:
        """
        Create from a pandas DataFrame.
        
        Expected columns: x, y, gene (and optionally transcript_id, count)
        """
        # Generate transcript IDs if not present
        if 'transcript_id' not in df.columns:
            df = df.copy()
            df['transcript_id'] = [f"tx_{i:08d}" for i in range(len(df))]
        
        transcripts = []
        for _, row in df.iterrows():
            transcripts.append(Transcript(
                transcript_id=row['transcript_id'],
                x=float(row['x']),
                y=float(row['y']),
                gene=str(row['gene']),
                count=int(row.get('count', 1)),
            ))
        
        return cls(transcripts)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = [t.to_dict() for t in self.transcripts]
        df = pd.DataFrame(data)
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_transcripts": len(self.transcripts),
            "transcripts": [t.to_dict() for t in self.transcripts],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptCollection:
        """Create from dictionary."""
        transcripts = [Transcript.from_dict(t) for t in data["transcripts"]]
        return cls(transcripts)
    
    def save_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> TranscriptCollection:
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_csv(self, filepath: str):
        """Save to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def get_assigned_transcripts(self) -> List[Transcript]:
        """Get only transcripts that have been assigned to a cell."""
        return [t for t in self.transcripts if t.cell_id is not None]
    
    def get_unassigned_transcripts(self) -> List[Transcript]:
        """Get transcripts not yet assigned to any cell."""
        return [t for t in self.transcripts if t.cell_id is None]
    
    def assign_to_cells(self, cell_polygons: Dict[str, Any]):
        """
        Assign transcripts to cells based on polygon containment.
        
        Args:
            cell_polygons: Dict mapping cell_id -> shapely Polygon
        """
        from shapely.geometry import Point
        
        for transcript in self.transcripts:
            point = Point(transcript.x, transcript.y)
            for cell_id, polygon in cell_polygons.items():
                if polygon.contains(point):
                    transcript.cell_id = cell_id
                    break
    
    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.transcripts:
            return {}
        
        genes = [t.gene for t in self.transcripts]
        assigned = len(self.get_assigned_transcripts())
        unassigned = len(self.get_unassigned_transcripts())
        
        return {
            "n_transcripts": len(self.transcripts),
            "n_unique_genes": len(set(genes)),
            "assignment_rate": assigned / len(self.transcripts) if self.transcripts else 0,
            "n_assigned": assigned,
            "n_unassigned": unassigned,
        }
