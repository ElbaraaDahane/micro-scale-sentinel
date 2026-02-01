import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Layer 4: Generates visual and textual reports from system data.
    """

    def __init__(self, storage_manager):
        self.storage = storage_manager

    def generate_summary_report(self) -> Dict[str, Any]:
        """Wraps storage stats calculation."""
        return self.storage.get_statistics()

    def create_visualizations(self, output_dir: str):
        """Generates static charts using Matplotlib."""
        stats = self.storage.get_statistics()
        if stats['total_particles'] == 0:
            logger.warning("No data to visualize.")
            return

        # 1. Classification Distribution
        plt.figure(figsize=(10, 6))
        classes = stats['classification_counts']
        plt.bar(classes.keys(), classes.values(), color=['red', 'green', 'gray'])
        plt.title('Distribution of Particle Classifications')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.savefig(f"{output_dir}/class_dist.png")
        plt.close()

        # 2. Polymer Types
        if 'polymer_counts' in stats and stats['polymer_counts']:
            plt.figure(figsize=(10, 6))
            polymers = stats['polymer_counts']
            plt.pie(polymers.values(), labels=polymers.keys(), autopct='%1.1f%%')
            plt.title('Detected Polymer Types')
            plt.savefig(f"{output_dir}/polymer_dist.png")
            plt.close()

    def export_pdf_report(self, output_path: str):
        """Generates a multipage PDF report containing stats and plots."""
        stats = self.storage.get_statistics()
        if stats['total_particles'] == 0:
            return

        with PdfPages(output_path) as pdf:
            # Page 1: Summary Text
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            txt = f"MICRO-SCALE SENTINEL REPORT\n\n"
            txt += f"Generated: {datetime.now().isoformat()}\n"
            txt += f"Total Particles Analyzed: {stats['total_particles']}\n"
            txt += f"Average Confidence: {stats.get('avg_confidence', 0):.2f}%\n\n"

            txt += "Breakdown:\n"
            for k, v in stats.get('classification_counts', {}).items():
                txt += f" - {k}: {v}\n"

            plt.text(0.1, 0.9, txt, fontsize=12, verticalalignment='top')
            pdf.savefig()
            plt.close()

            # Page 2: Visuals
            # We recreate plots here to save into PDF context
            plt.figure(figsize=(10, 6))
            classes = stats.get('classification_counts', {})
            if classes:
                plt.bar(classes.keys(), classes.values(), color=['#ff9999', '#66b3ff', '#99ff99'])
                plt.title('Classification Distribution')
                pdf.savefig()
                plt.close()

            # Page 3: Polymer Types
            polymers = stats.get('polymer_counts', {})
            if polymers:
                plt.figure(figsize=(10, 6))
                plt.pie(polymers.values(), labels=polymers.keys(), autopct='%1.1f%%')
                plt.title('Polymer Composition')
                pdf.savefig()
                plt.close()

        logger.info(f"PDF Report generated at {output_path}")

    def create_dashboard_data(self) -> str:
        """Prepares JSON data for external dashboard consumption."""
        stats = self.storage.get_statistics()
        return json.dumps(stats, indent=2)