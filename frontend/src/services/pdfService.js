import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import {
  getClinicalDisplayState,
  buildPdfAnalysisPayload,
} from '../utils/clinicalPriority';

export class PDFReportService {
  constructor() {
    this.doc = null;
    this.pageWidth = 210; // mm (A4)
    this.pageHeight = 297; // mm (A4)
    this.margin = 20; // mm
    this.contentWidth = this.pageWidth - 2 * this.margin;
  }

  async generateAnalysisReport(analysisData, imageData, clinicalState = null) {
    this.doc = new jsPDF('p', 'mm', 'a4');
    const payload = buildPdfAnalysisPayload(analysisData) ?? analysisData;
    this._clinicalState = clinicalState ?? getClinicalDisplayState(payload);

    // Configuration des polices
    this.doc.setFont('helvetica');
    
    let yPosition = this.margin;
    
    // 1. En-tête avec logo et titre
    yPosition = await this.addHeader(yPosition);
    
    // 2. Informations du patient et de l'analyse
    yPosition = this.addPatientInfo(analysisData, yPosition);
    
    // 3. Image radiographique
    yPosition = await this.addRadiographyImage(imageData, yPosition);
    
    // 4. Résultats principaux
    yPosition = this.addMainFindings(analysisData, yPosition);
    
    // 5. Tableau détaillé des pathologies
    yPosition = this.addDetailedPathologies(analysisData, yPosition);
    
    // 6. Pied de page
    this.addFooter();
    
    return this.doc;
  }

  async addHeader(yPosition) {
    // Titre RADIOX centré
    this.doc.setFontSize(28);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setTextColor(0, 212, 255);
    const radioXText = 'RADIOX';
    const radioXWidth = this.doc.getTextWidth(radioXText);
    const radioXX = (this.pageWidth - radioXWidth) / 2;
    this.doc.text(radioXText, radioXX, yPosition);
    
    yPosition += 12;
    
    // Sous-titre Rapport d'Analyse Radiographique centré
    this.doc.setFontSize(18);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setTextColor(50);
    const subtitleText = 'Rapport d\'Analyse Radiographique';
    const subtitleWidth = this.doc.getTextWidth(subtitleText);
    const subtitleX = (this.pageWidth - subtitleWidth) / 2;
    this.doc.text(subtitleText, subtitleX, yPosition);
    
    yPosition += 15;
    
    // Ligne de séparation
    this.doc.setDrawColor(0, 212, 255);
    this.doc.setLineWidth(0.5);
    this.doc.line(this.margin, yPosition, this.pageWidth - this.margin, yPosition);
    
    yPosition += 10;
    
    // Informations sur le modèle
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'normal');
    this.doc.text('Modèle IA: DenseNet-121 NIH ChestX-ray14', this.margin, yPosition);
    yPosition += 5;
    this.doc.text('Date d\'analyse: ' + new Date().toLocaleDateString('fr-FR'), this.margin, yPosition);
    
    yPosition += 10;
    return yPosition;
  }

  addPatientInfo(analysisData, yPosition) {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Informations du Patient', this.margin, yPosition);
    
    yPosition += 8;
    
    this.doc.setFontSize(10);
    this.doc.setFont('helvetica', 'normal');
    
    const { clinical } = this._clinicalState;
    const primaryPct = clinical.primary
      ? Math.round((clinical.primary.confidence ?? clinical.primary.probability ?? 0) * 100)
      : (analysisData.confidence || 0);

    const patientInfo = [
      `ID Patient: ${analysisData.patientId || 'N/A'}`,
      `ID Analyse: ${analysisData.analysis_id || 'N/A'}`,
      `Date: ${analysisData.date || new Date().toLocaleDateString('fr-FR')}`,
      `Conclusion principale: ${clinical.primary?.pathology || 'N/A'} (${primaryPct}%)`
    ];
    
    patientInfo.forEach(info => {
      this.doc.text(info, this.margin + 5, yPosition);
      yPosition += 5;
    });
    
    yPosition += 10;
    return yPosition;
  }

  async addRadiographyImage(imageData, yPosition) {
    if (!imageData) return yPosition;
    
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Images Radiographiques', this.margin, yPosition);
    
    yPosition += 8;
    
    try {
      // Image originale à gauche
      const imgWidth = 70; // mm (plus petit pour deux images)
      const imgHeight = 70; // mm (plus petit pour deux images)
      const imgLeftX = this.margin + 10; // Marge gauche
      
      // Utiliser image_base64 si disponible
      const imageSource = imageData.image_base64 || imageData;
      this.doc.addImage(imageSource, 'JPEG', imgLeftX, yPosition, imgWidth, imgHeight);
      
      // Image avec Grad-CAM superposée à droite
      if (imageData.heatmap_base64) {
        const imgRightX = this.margin + 90; // Position droite
        
        // Créer l'image composite dans le frontend
        const compositeImage = await this.createCompositeImage(imageSource, imageData.heatmap_base64);
        
        // Légende pour l'image originale
        this.doc.setFontSize(9);
        this.doc.setFont('helvetica', 'normal');
        this.doc.text('Image originale', imgLeftX + 5, yPosition + imgHeight + 5);
        
        // Légende pour l'image avec Grad-CAM
        this.doc.text('Image avec Grad-CAM', imgRightX + 5, yPosition + imgHeight + 5);
        
        // Ajouter l'image composite (originale + Grad-CAM)
        this.doc.addImage(compositeImage, 'JPEG', imgRightX, yPosition, imgWidth, imgHeight);
        
        yPosition += imgHeight + 15; // Espace supplémentaire pour les légendes
      } else {
        // Si pas de Grad-CAM, centrer l'image seule
        const centerX = (this.pageWidth - imgWidth) / 2;
        this.doc.addImage(imageSource, 'JPEG', centerX, yPosition, imgWidth, imgHeight);
        yPosition += imgHeight + 10;
      }
      
    } catch (error) {
      console.warn('Impossible d\'ajouter les images au PDF:', error);
      yPosition += 10;
    }
    
    return yPosition;
  }

  async createCompositeImage(originalImageBase64, heatmapBase64) {
    return new Promise((resolve, reject) => {
      try {
        console.log("=== COMPOSITE DEBUG ===");
        console.log("Original image base64 length:", originalImageBase64.length);
        console.log("Heatmap base64 length:", heatmapBase64.length);
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Créer un canvas plus grand pour éviter la compression
        canvas.width = 500;
        canvas.height = 500;
        
        const originalImg = new Image();
        const heatmapImg = new Image();
        
        originalImg.onload = () => {
          console.log("Original image loaded successfully");
          // Dessiner l'image originale à grande taille
          ctx.drawImage(originalImg, 0, 0, 500, 500);
          
          heatmapImg.onload = () => {
            console.log("Heatmap loaded successfully");
            // Dessiner la Grad-CAM par-dessus avec transparence
            ctx.globalAlpha = 0.7; // 70% d'opacité pour plus de visibilité
            ctx.drawImage(heatmapImg, 0, 0, 500, 500);
            ctx.globalAlpha = 1.0;
            
            // Convertir en PNG pour éviter la compression JPEG
            const compositeBase64 = canvas.toDataURL('image/png');
            console.log("Composite created successfully, length:", compositeBase64.length);
            resolve(compositeBase64);
          };
          
          heatmapImg.onerror = (error) => {
            console.error('Failed to load heatmap:', error);
            reject(new Error('Failed to load heatmap'));
          };
          heatmapImg.src = `data:image/png;base64,${heatmapBase64}`;
        };
        
        originalImg.onerror = (error) => {
          console.error('Failed to load original image:', error);
          reject(new Error('Failed to load original image'));
        };
        originalImg.src = `data:image/jpeg;base64,${originalImageBase64}`;
      } catch (error) {
        console.error('Error in createCompositeImage:', error);
        reject(error);
      }
    });
  }

  addMainFindings(analysisData, yPosition) {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Conclusion clinique principale', this.margin, yPosition);

    yPosition += 8;
    const { clinical } = this._clinicalState;

    if (clinical.primary) {
      this.doc.setFontSize(11);
      this.doc.setFont('helvetica', 'bold');
      this.doc.text('Conclusion clinique :', this.margin + 5, yPosition);
      yPosition += 6;

      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'normal');
      this.doc.text(clinical.conclusionLine, this.margin + 5, yPosition);
      yPosition += 6;

      const confidence = clinical.primary.confidence ?? clinical.primary.probability ?? 0;
      const confidenceColor = this.getConfidenceColor(confidence);
      this.doc.setTextColor(confidenceColor.r, confidenceColor.g, confidenceColor.b);
      this.doc.text(
        `Niveau de risque : ${this.getSeverityLabel(clinical.primary.severity)}`,
        this.margin + 5,
        yPosition
      );
      this.doc.setTextColor(0, 0, 0);
      yPosition += 8;

      if (clinical.contributors.length > 0) {
        this.doc.setFontSize(10);
        this.doc.setFont('helvetica', 'bold');
        this.doc.text('Anomalies contributrices :', this.margin + 5, yPosition);
        yPosition += 6;

        this.doc.setFont('helvetica', 'normal');
        clinical.contributors.forEach((contributor) => {
          if (yPosition > this.pageHeight - 30) {
            this.doc.addPage();
            yPosition = this.margin;
          }
          const line = `- ${contributor.pathology} (${Math.round(contributor.probability * 100)}%)`;
          this.doc.text(line, this.margin + 8, yPosition);
          yPosition += 5;
        });
        yPosition += 4;
      }
    } else {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text('Aucune conclusion clinique disponible', this.margin + 5, yPosition);
      yPosition += 8;
    }

    return yPosition;
  }

  addDetailedPathologies(analysisData, yPosition) {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Analyse Détaillée des Pathologies', this.margin, yPosition);
    
    yPosition += 8;
    
    const { displayPredictions } = this._clinicalState;

    if (displayPredictions.length > 0) {
      // En-tête du tableau
      const headers = ['Pathologie', 'Confiance', 'Risque'];
      const columnWidths = [80, 40, 40];
      let xPos = this.margin;
      
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'bold');
      headers.forEach((header, index) => {
        this.doc.text(header, xPos, yPosition);
        xPos += columnWidths[index];
      });
      
      yPosition += 6;
      
      // Ligne de séparation
      this.doc.setDrawColor(200);
      this.doc.line(this.margin, yPosition, this.pageWidth - this.margin, yPosition);
      yPosition += 4;
      
      // Données du tableau (même logique d'affichage que l'écran résultats)
      this.doc.setFont('helvetica', 'normal');
      const sortedPredictions = [...displayPredictions].sort(
        (a, b) => b.probability - a.probability
      );
      
      sortedPredictions.forEach((prediction, index) => {
        if (yPosition > this.pageHeight - 30) {
          this.doc.addPage();
          yPosition = this.margin;
        }
        
        xPos = this.margin;
        
        // Pathologie
        this.doc.text(prediction.pathology, xPos, yPosition);
        xPos += columnWidths[0];
        
        // Confiance
        const confidence = Math.round(prediction.probability * 100);
        this.doc.text(`${confidence}%`, xPos, yPosition);
        xPos += columnWidths[1];
        
        // Risque avec couleur
        const severityColor = this.getSeverityColor(prediction.severity);
        this.doc.setTextColor(severityColor.r, severityColor.g, severityColor.b);
        this.doc.text(this.getSeverityLabel(prediction.severity), xPos, yPosition);
        this.doc.setTextColor(0, 0, 0); // Reset couleur
        
        yPosition += 5;
      });
    } else {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text('Aucune pathologie détectée', this.margin + 5, yPosition);
      yPosition += 8;
    }
    
    yPosition += 10;
    return yPosition;
  }

  addFooter() {
    const footerY = this.pageHeight - 25;
    const lineHeight = 5;
    
    this.doc.setFontSize(8);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setTextColor(100);
    
    // Ligne au-dessus du footer
    this.doc.setDrawColor(200);
    this.doc.line(this.margin, footerY - 10, this.pageWidth - this.margin, footerY - 10);
    
    // Footer multi-lignes comme dans l'impression
    const footerLines = [
      'RADIOX - Système d\'Analyse Radiographique par IA',
      'Ce rapport a été généré automatiquement par RadioX - Les résultats doivent être confirmés par un professionnel de santé qualifié.',
      `Modèle: DenseNet-121 NIH ChestX-ray14 | Date de génération: ${new Date().toLocaleString('fr-FR')}`
    ];
    
    footerLines.forEach((line, index) => {
      const y = footerY + (index * lineHeight);
      const textWidth = this.doc.getTextWidth(line);
      const xPosition = (this.pageWidth - textWidth) / 2;
      
      if (index === 0) {
        this.doc.setFont('helvetica', 'bold');
      } else {
        this.doc.setFont('helvetica', 'normal');
      }
      
      this.doc.text(line, xPosition, y);
    });
  }

  getConfidenceColor(probability) {
    if (probability >= 0.8) return { r: 0, g: 255, b: 200 }; // Vert cyan
    if (probability >= 0.6) return { r: 255, g: 211, b: 0 }; // Jaune
    return { r: 255, g: 71, b: 87 }; // Rouge
  }

  getSeverityColor(severity) {
    switch (severity) {
      case 'high': return { r: 255, g: 71, b: 87 }; // Rouge
      case 'moderate': return { r: 255, g: 159, b: 67 }; // Orange
      default: return { r: 0, g: 255, b: 200 }; // Vert cyan
    }
  }

  getSeverityLabel(severity) {
    switch (severity) {
      case 'high': return 'Élevé';
      case 'moderate': return 'Modéré';
      default: return 'Faible';
    }
  }

  async exportToPDF(analysisData, imageData, filename = null, options = {}) {
    try {
      const payload = buildPdfAnalysisPayload(analysisData) ?? analysisData;
      const clinicalState =
        options.clinicalState ?? getClinicalDisplayState(payload);
      const pdf = await this.generateAnalysisReport(
        payload,
        imageData,
        clinicalState
      );
      const defaultFilename = `RadioX_Rapport_${new Date().toISOString().split('T')[0]}.pdf`;
      const finalFilename = filename || defaultFilename;
      
      pdf.save(finalFilename);
      return { success: true, filename: finalFilename };
    } catch (error) {
      console.error('Erreur lors de la génération du PDF:', error);
      return { success: false, error: error.message };
    }
  }
}

export default new PDFReportService();
