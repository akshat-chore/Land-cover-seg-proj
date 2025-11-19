# Dashboard Testing & Usage Guide

## Quick Start

### 1. Start the Backend API
```bash
cd d:\land-cover-seg\Land-Cover-Semantic-Segmentation-PyTorch
python -m uvicorn app.main:app --reload
```

### 2. Open the Dashboard
```
Open dashboard.html in your web browser
OR
Navigate to http://localhost:8000/docs (if serving the HTML)
```

---

## Features to Test

### ✅ Image Upload & Inference Progress

**Test Steps:**
1. Click "Upload Image" or drag an image file
2. Select a satellite/aerial image (JPG, PNG, TIFF)
3. Image preview should appear
4. Progress bar shows with:
   - Animated fill (0% → 100%)
   - Percentage display
   - Estimated time remaining countdown

**Expected Behavior:**
- Progress bar smoothly fills from left to right
- Percentage updates in real-time
- Time estimate decreases as processing completes
- Results display after bar reaches 100%

### ✅ Segmentation Results Display

**Expected Display:**
- Inference Time (in seconds)
- Image Size (width × height × channels)
- Segmentation Mask Size
- Classes Detected (list)
- Segmentation Mask Image
- Overlay on Original Image
- Per-Class Statistics (pixels, percentages, km²)

**Available Buttons:**
- 📥 Download Results → Downloads JSON file
- 📄 Generate AI Report → Triggers report generation

### ✅ AI Report Generation

**Test Steps:**
1. Complete image upload & processing
2. Scroll down or look for "Generate AI Report" button
3. Click "Generate AI Report"
4. Report panel should appear at bottom with progress bar
5. Watch progress bar fill (0% → 100%) over ~30 seconds
6. Report cards display when complete

**Expected Progress Bar:**
- Shows "⏳ Generating Report with Gemini AI"
- Percentage and time estimate display
- Smooth animation
- Disappears when report is ready

### ✅ Persistent Report Panel

**Test What Happens:**
1. Report displays in a persistent panel (NOT a modal)
2. Report doesn't cover the segmentation results above
3. Report content shows in color-coded cards:
   - 📋 Executive Summary (Blue)
   - 🏗️ Urban Planning (Orange)
   - 🚨 Disaster Management (Red)
   - ⚙️ Automation & Accuracy (Green)
   - 💡 Recommendations (Purple)

**Card Content:**
- Executive Summary: 3-4 sentences
- Sections with bullet points:
  - Urban Planning insights (2+ items)
  - Disaster Management insights (2+ items)
  - Automation accuracy assessment (2+ items)
  - Model Improvements (list)
  - Deployment Notes (list)

### ✅ Report Hide/Show Control

**Test Steps:**
1. Generate a report (report panel shows)
2. Click "✕ Hide Report" button (top right of report header)
3. Report panel should disappear/hide
4. Scroll up to see results section
5. **Reload page OR click "Generate Report" again**
6. Report should still be there OR regenerate with same data
7. Report data should NOT require clicking upload again

**Expected Behavior:**
- Hiding report doesn't delete the data
- Report can be shown again without regenerating
- User can see both results and report together

### ✅ Responsive Design

**Test on Different Screen Sizes:**

Desktop (1200px+):
- Report in 2-column grid
- 2 cards per row
- Executive summary spans full width

Tablet (768px - 1199px):
- Report in 1-2 column layout
- Smooth transition
- Cards still visible

Mobile (< 768px):
- Report in 1-column layout
- Cards stack vertically
- Full width cards
- Scrollable content

### ✅ Error Handling

**Test Error Cases:**

1. **No Image Upload:**
   - Click "Generate AI Report" without uploading image
   - Expected: Error message "No prediction data available..."

2. **API Connection Error:**
   - Close backend API
   - Try to upload image
   - Expected: Error message with status code

3. **Invalid Image:**
   - Upload a non-image file
   - Expected: Error message from API

---

## Visual Verification Checklist

- [ ] Progress bar appears during image processing
- [ ] Progress bar percentage increments smoothly
- [ ] Time remaining updates correctly
- [ ] Progress bar disappears after completion
- [ ] Results section displays all expected data
- [ ] Report panel appears full-width (not modal)
- [ ] Report panel has "Hide Report" button
- [ ] Color-coded section headers visible
- [ ] Report sections have bullet points
- [ ] Mobile layout collapses to single column
- [ ] Scroll works smoothly with both results and report
- [ ] Hide button hides the report
- [ ] Report data persists (doesn't require regeneration)

---

## Common Issues & Solutions

### Issue: Progress bar doesn't appear
**Solution:** Check browser console (F12) for JavaScript errors

### Issue: Report panel overlaps content (modal-like)
**Solution:** Verify CSS classes are applied correctly - check page CSS

### Issue: Report shows no content
**Solution:** 
- Check backend API is running
- Verify /report endpoint returns proper JSON
- Check browser console for API error messages

### Issue: Progress bar stuck at 0%
**Solution:**
- Check JavaScript in page source
- Verify `startProgressSimulation()` is being called
- Look for console errors

### Issue: Report appears as modal overlay
**Solution:**
- This shouldn't happen - verify CSS was updated correctly
- Check that `displayReportModal()` is not being called
- Ensure `displayReportPanel()` is being used instead

---

## Browser Developer Tools

### To Debug:
1. Open Developer Tools (F12 or Right-click → Inspect)
2. Go to "Console" tab
3. Look for any red errors
4. Check "Network" tab to see API calls
5. Check "Elements" tab to verify CSS classes

### To Test API Calls:
```javascript
// In browser console, test the report endpoint:
fetch('http://127.0.0.1:8000/report', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        metrics_json: {total_pixels: 262144},
        segmentation_summary: {image_shape: [512, 512, 3]},
        context: {analysis_type: 'test'}
    })
})
.then(r => r.json())
.then(d => console.log(d))
```

---

## Performance Notes

### Expected Timings:
- Image Upload & Processing: 5-15 seconds
- Report Generation: 20-40 seconds
- Report Rendering: < 1 second
- Page Load: < 1 second

### Progress Bar Accuracy:
- Simulated (not real)
- Based on assumed duration
- Will reach 95% and wait for actual response
- Completes to 100% when data arrives

---

## File References

- **Dashboard HTML:** `dashboard.html`
- **Backend API:** `app/main.py` (port 8000)
- **Report Endpoint:** `app/main.py` → `/report`
- **Gemini Client:** `app/gemini_client.py`

---

## Support

If you encounter any issues:

1. Check console for errors: F12 → Console
2. Verify backend is running: `http://127.0.0.1:8000/docs`
3. Check API response format matches expectations
4. Review `FRONTEND_IMPROVEMENTS.md` for detailed changes
5. Check `FRONTEND_VISUAL_GUIDE.md` for layout reference

---

## Version Info

- **Dashboard Version:** 2.1 (Updated with progress bars and persistent reports)
- **Date:** November 11, 2025
- **Tested On:** Chrome, Firefox, Edge (Modern versions)
- **Backend:** FastAPI with Gemini Integration
