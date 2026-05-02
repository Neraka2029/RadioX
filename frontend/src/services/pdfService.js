import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export class PDFReportService {
  constructor() {
    this.doc = null;
    this.pageWidth = 210; // mm (A4)
    this.pageHeight = 297; // mm (A4)
    this.margin = 20; // mm
    this.contentWidth = this.pageWidth - 2 * this.margin;
  }

  async generateAnalysisReport(analysisData, imageData) {
    this.doc = new jsPDF('p', 'mm', 'a4');
    
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
    
    // 6. Recommandations
    yPosition = this.addRecommendations(analysisData, yPosition);
    
    // 7. Pied de page
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
    
    const patientInfo = [
      `ID Patient: ${analysisData.patientId || 'N/A'}`,
      `ID Analyse: ${analysisData.analysis_id || 'N/A'}`,
      `Date: ${analysisData.date || new Date().toLocaleDateString('fr-FR')}`,
      `Confiance: ${analysisData.confidence || 0}%`
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
    this.doc.text('Résultats Principaux', this.margin, yPosition);
    
    yPosition += 8;
    
    if (analysisData.predictions && analysisData.predictions.length > 0) {
      // Trouver la prédiction principale
      const mainPrediction = analysisData.predictions.reduce((a, b) => 
        a.probability > b.probability ? a : b
      );
      
      this.doc.setFontSize(11);
      this.doc.setFont('helvetica', 'bold');
      this.doc.text(`Pathologie principale: ${mainPrediction.pathology}`, this.margin + 5, yPosition);
      
      yPosition += 6;
      
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'normal');
      this.doc.text(`Confiance: ${Math.round(mainPrediction.probability * 100)}%`, this.margin + 5, yPosition);
      
      yPosition += 6;
      
      // Ajouter la couleur de confiance
      const confidenceColor = this.getConfidenceColor(mainPrediction.probability);
      this.doc.setTextColor(confidenceColor.r, confidenceColor.g, confidenceColor.b);
      this.doc.text(`Niveau de risque: ${this.getSeverityLabel(mainPrediction.severity)}`, this.margin + 5, yPosition);
      this.doc.setTextColor(0, 0, 0); // Reset couleur
      
      yPosition += 8;
    } else {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'italic');
      this.doc.text('Aucune prédiction disponible', this.margin + 5, yPosition);
      yPosition += 8;
    }
    
    return yPosition;
  }

  addDetailedPathologies(analysisData, yPosition) {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Analyse Détaillée des Pathologies', this.margin, yPosition);
    
    yPosition += 8;
    
    if (analysisData.predictions && analysisData.predictions.length > 0) {
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
      
      // Données du tableau
      this.doc.setFont('helvetica', 'normal');
      const sortedPredictions = analysisData.predictions.sort((a, b) => b.probability - a.probability);
      
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

  addRecommendations(analysisData, yPosition) {
    this.doc.setFontSize(12);
    this.doc.setFont('helvetica', 'bold');
    this.doc.text('Recommandations Cliniques', this.margin, yPosition);
    
    yPosition += 8;
    
    if (analysisData.recommendations && analysisData.recommendations.length > 0) {
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'normal');
      
      analysisData.recommendations.forEach((recommendation, index) => {
        if (yPosition > this.pageHeight - 30) {
          this.doc.addPage();
          yPosition = this.margin;
        }
        
        // Ajouter un point pour chaque recommandation
        this.doc.text(`${index + 1}. ${recommendation}`, this.margin + 5, yPosition);
        yPosition += 6;
      });
    } else {
      // Recommandations par défaut selon les résultats
      const defaultRecommendations = this.getDefaultRecommendations(analysisData);
      
      this.doc.setFontSize(10);
      this.doc.setFont('helvetica', 'normal');
      
      defaultRecommendations.forEach((recommendation, index) => {
        if (yPosition > this.pageHeight - 30) {
          this.doc.addPage();
          yPosition = this.margin;
        }
        
        this.doc.text(`${index + 1}. ${recommendation}`, this.margin + 5, yPosition);
        yPosition += 6;
      });
    }
    
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

  getDefaultRecommendations(analysisData) {
    const recommendations = [
      'Les résultats de cette analyse IA doivent être interprétés par un radiologue qualifié.',
      'Une corrélation clinique avec les symptômes du patient est essentielle.',
      'En cas de doute, des examens complémentaires peuvent être nécessaires.'
    ];
    
    if (analysisData.predictions && analysisData.predictions.length > 0) {
      const highRiskPathologies = analysisData.predictions.filter(p => 
        p.probability > 0.5 && p.pathology !== 'Normal'
      );
      
      if (highRiskPathologies.length > 0) {
        recommendations.unshift('Pathologies à haut risque détectées - Évaluation clinique urgente recommandée.');
      }
    }
    
    return recommendations;
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

  async exportToPDF(analysisData, imageData, filename = null) {
    try {
      const pdf = await this.generateAnalysisReport(analysisData, imageData);
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
