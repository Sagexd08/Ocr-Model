"""
Evaluation report generator for CurioScan.

Generates comprehensive PDF reports with metrics, visualizations, and analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("model", {}).get("type", "renderer_classifier")
    
    def generate_report(self, results: Dict[str, Any], output_dir: Path,
                       dataset_name: str = "test", model_path: str = "") -> Path:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            dataset_name: Name of evaluated dataset
            model_path: Path to model checkpoint
            
        Returns:
            Path to generated report
        """
        
        logger.info("Generating evaluation report...")
        
        # Create report content
        report_content = self._create_report_content(
            results, dataset_name, model_path
        )
        
        # Generate HTML report
        html_report_path = self._generate_html_report(report_content, output_dir)
        
        # Try to generate PDF report
        pdf_report_path = self._generate_pdf_report(html_report_path, output_dir)
        
        logger.info(f"Report generated: {pdf_report_path or html_report_path}")
        
        return pdf_report_path or html_report_path
    
    def _create_report_content(self, results: Dict[str, Any], 
                              dataset_name: str, model_path: str) -> Dict[str, Any]:
        """Create structured report content."""
        
        # Extract key metrics
        overall_metrics = results.get("overall_metrics", {})
        confidence_metrics = results.get("confidence_metrics", {})
        per_class_metrics = results.get("per_class_metrics", {})
        error_analysis = results.get("error_analysis", {})
        processing_time = results.get("processing_time", {})
        
        # Create report structure
        report_content = {
            "metadata": {
                "title": "CurioScan Model Evaluation Report",
                "generated_at": datetime.now().isoformat(),
                "model_type": self.model_type,
                "dataset": dataset_name,
                "model_path": model_path,
                "config": self.config
            },
            "executive_summary": self._create_executive_summary(overall_metrics, processing_time),
            "detailed_metrics": {
                "overall_performance": overall_metrics,
                "confidence_analysis": confidence_metrics,
                "per_class_performance": per_class_metrics,
                "error_analysis": error_analysis,
                "performance_metrics": processing_time
            },
            "analysis": self._create_analysis_section(results),
            "recommendations": self._create_recommendations(results),
            "appendix": {
                "configuration": self.config,
                "raw_results": results
            }
        }
        
        return report_content
    
    def _create_executive_summary(self, overall_metrics: Dict[str, float],
                                 processing_time: Dict[str, float]) -> Dict[str, Any]:
        """Create executive summary section."""
        
        # Determine overall performance level
        f1_score = overall_metrics.get("f1_score", 0.0)
        
        if f1_score >= 0.95:
            performance_level = "Excellent"
        elif f1_score >= 0.90:
            performance_level = "Very Good"
        elif f1_score >= 0.80:
            performance_level = "Good"
        elif f1_score >= 0.70:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        # Key findings
        key_findings = []
        
        if overall_metrics.get("accuracy", 0.0) >= 0.95:
            key_findings.append("High accuracy achieved across all classes")
        
        if processing_time.get("avg_time_per_sample", 0.0) < 2.0:
            key_findings.append("Fast inference time suitable for production")
        
        if overall_metrics.get("f1_score", 0.0) - overall_metrics.get("accuracy", 0.0) > 0.05:
            key_findings.append("Model shows good balance between precision and recall")
        
        return {
            "performance_level": performance_level,
            "key_metrics": {
                "accuracy": overall_metrics.get("accuracy", 0.0),
                "f1_score": overall_metrics.get("f1_score", 0.0),
                "precision": overall_metrics.get("precision", 0.0),
                "recall": overall_metrics.get("recall", 0.0)
            },
            "performance_summary": {
                "throughput": processing_time.get("throughput", 0.0),
                "avg_time_per_sample": processing_time.get("avg_time_per_sample", 0.0),
                "total_samples": processing_time.get("sample_count", 0)
            },
            "key_findings": key_findings
        }
    
    def _create_analysis_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed analysis section."""
        
        analysis = {}
        
        # Performance analysis
        overall_metrics = results.get("overall_metrics", {})
        confidence_metrics = results.get("confidence_metrics", {})
        
        analysis["performance_analysis"] = {
            "strengths": [],
            "weaknesses": [],
            "observations": []
        }
        
        # Identify strengths
        if overall_metrics.get("accuracy", 0.0) >= 0.90:
            analysis["performance_analysis"]["strengths"].append("High overall accuracy")
        
        if confidence_metrics.get("calibration_error", 1.0) < 0.1:
            analysis["performance_analysis"]["strengths"].append("Well-calibrated confidence scores")
        
        # Identify weaknesses
        error_analysis = results.get("error_analysis", {})
        if error_analysis.get("high_confidence_errors", 0) > 0:
            analysis["performance_analysis"]["weaknesses"].append("Some high-confidence errors present")
        
        # Class-specific analysis
        per_class_metrics = results.get("per_class_metrics", {})
        if per_class_metrics:
            class_analysis = self._analyze_class_performance(per_class_metrics)
            analysis["class_performance_analysis"] = class_analysis
        
        # Confidence analysis
        if confidence_metrics:
            conf_analysis = self._analyze_confidence_behavior(confidence_metrics)
            analysis["confidence_analysis"] = conf_analysis
        
        return analysis
    
    def _analyze_class_performance(self, per_class_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze per-class performance."""
        
        # Find best and worst performing classes
        class_f1_scores = {
            class_name: metrics.get("f1", 0.0)
            for class_name, metrics in per_class_metrics.items()
        }
        
        best_class = max(class_f1_scores, key=class_f1_scores.get)
        worst_class = min(class_f1_scores, key=class_f1_scores.get)
        
        # Calculate performance variance
        f1_scores = list(class_f1_scores.values())
        performance_variance = max(f1_scores) - min(f1_scores)
        
        return {
            "best_performing_class": {
                "class": best_class,
                "f1_score": class_f1_scores[best_class]
            },
            "worst_performing_class": {
                "class": worst_class,
                "f1_score": class_f1_scores[worst_class]
            },
            "performance_variance": performance_variance,
            "balanced_performance": performance_variance < 0.1
        }
    
    def _analyze_confidence_behavior(self, confidence_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze confidence behavior."""
        
        analysis = {
            "calibration_quality": "good" if confidence_metrics.get("calibration_error", 1.0) < 0.1 else "poor",
            "confidence_level": "high" if confidence_metrics.get("avg_confidence", 0.0) > 0.8 else "moderate",
            "reliability_at_high_confidence": confidence_metrics.get("accuracy_at_0.8", 0.0)
        }
        
        return analysis
    
    def _create_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Create recommendations based on results."""
        
        recommendations = []
        
        overall_metrics = results.get("overall_metrics", {})
        confidence_metrics = results.get("confidence_metrics", {})
        error_analysis = results.get("error_analysis", {})
        
        # Performance-based recommendations
        if overall_metrics.get("f1_score", 0.0) < 0.90:
            recommendations.append("Consider additional training data or model architecture improvements")
        
        if confidence_metrics.get("calibration_error", 0.0) > 0.1:
            recommendations.append("Implement confidence calibration techniques")
        
        if error_analysis.get("high_confidence_errors", 0) > 0:
            recommendations.append("Review high-confidence errors for systematic issues")
        
        # Class-specific recommendations
        per_class_metrics = results.get("per_class_metrics", {})
        if per_class_metrics:
            worst_classes = [
                class_name for class_name, metrics in per_class_metrics.items()
                if metrics.get("f1", 0.0) < 0.80
            ]
            
            if worst_classes:
                recommendations.append(f"Focus on improving performance for: {', '.join(worst_classes)}")
        
        # Processing time recommendations
        processing_time = results.get("processing_time", {})
        if processing_time.get("avg_time_per_sample", 0.0) > 5.0:
            recommendations.append("Consider model optimization for faster inference")
        
        return recommendations
    
    def _generate_html_report(self, report_content: Dict[str, Any], output_dir: Path) -> Path:
        """Generate HTML report."""
        
        html_template = self._get_html_template()
        
        # Fill template with content
        html_content = html_template.format(
            title=report_content["metadata"]["title"],
            generated_at=report_content["metadata"]["generated_at"],
            model_type=report_content["metadata"]["model_type"],
            dataset=report_content["metadata"]["dataset"],
            executive_summary=self._format_executive_summary(report_content["executive_summary"]),
            detailed_metrics=self._format_detailed_metrics(report_content["detailed_metrics"]),
            analysis=self._format_analysis(report_content["analysis"]),
            recommendations=self._format_recommendations(report_content["recommendations"])
        )
        
        # Save HTML report
        html_path = output_dir / "evaluation_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_pdf_report(self, html_path: Path, output_dir: Path) -> Optional[Path]:
        """Generate PDF report from HTML."""
        
        try:
            import pdfkit
            
            pdf_path = output_dir / "evaluation_report.pdf"
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            pdfkit.from_file(str(html_path), str(pdf_path), options=options)
            
            return pdf_path
            
        except ImportError:
            logger.warning("pdfkit not available, PDF report not generated")
            return None
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return None
    
    def _get_html_template(self) -> str:
        """Get HTML template for report."""
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin: 10px 0; }}
                .metric-value {{ font-weight: bold; color: #2E86AB; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ margin: 5px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #2E86AB; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated: {generated_at}</p>
                <p>Model Type: {model_type} | Dataset: {dataset}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {executive_summary}
            </div>
            
            <div class="section">
                <h2>Detailed Metrics</h2>
                {detailed_metrics}
            </div>
            
            <div class="section">
                <h2>Analysis</h2>
                {analysis}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations}
            </div>
        </body>
        </html>
        """
    
    def _format_executive_summary(self, summary: Dict[str, Any]) -> str:
        """Format executive summary as HTML."""
        
        key_metrics = summary["key_metrics"]
        performance_summary = summary["performance_summary"]
        
        html = f"""
        <div class="metric">Performance Level: <span class="metric-value">{summary["performance_level"]}</span></div>
        <div class="metric">Accuracy: <span class="metric-value">{key_metrics["accuracy"]:.3f}</span></div>
        <div class="metric">F1 Score: <span class="metric-value">{key_metrics["f1_score"]:.3f}</span></div>
        <div class="metric">Throughput: <span class="metric-value">{performance_summary["throughput"]:.1f} samples/sec</span></div>
        
        <h3>Key Findings</h3>
        <ul>
        """
        
        for finding in summary["key_findings"]:
            html += f"<li>{finding}</li>"
        
        html += "</ul>"
        
        return html
    
    def _format_detailed_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format detailed metrics as HTML."""
        
        html = "<h3>Overall Performance</h3>"
        
        overall = metrics["overall_performance"]
        html += f"""
        <div class="metric">Accuracy: <span class="metric-value">{overall.get("accuracy", 0.0):.3f}</span></div>
        <div class="metric">Precision: <span class="metric-value">{overall.get("precision", 0.0):.3f}</span></div>
        <div class="metric">Recall: <span class="metric-value">{overall.get("recall", 0.0):.3f}</span></div>
        <div class="metric">F1 Score: <span class="metric-value">{overall.get("f1_score", 0.0):.3f}</span></div>
        """
        
        return html
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as HTML."""
        
        html = ""
        
        if "performance_analysis" in analysis:
            perf_analysis = analysis["performance_analysis"]
            
            html += "<h3>Performance Analysis</h3>"
            
            if perf_analysis["strengths"]:
                html += "<h4>Strengths</h4><ul>"
                for strength in perf_analysis["strengths"]:
                    html += f"<li>{strength}</li>"
                html += "</ul>"
            
            if perf_analysis["weaknesses"]:
                html += "<h4>Areas for Improvement</h4><ul>"
                for weakness in perf_analysis["weaknesses"]:
                    html += f"<li>{weakness}</li>"
                html += "</ul>"
        
        return html
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML."""
        
        html = ""
        
        for i, recommendation in enumerate(recommendations, 1):
            html += f'<div class="recommendation">{i}. {recommendation}</div>'
        
        return html
