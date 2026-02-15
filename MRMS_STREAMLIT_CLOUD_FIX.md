MRMS decode still failing with: "unrecognized engine 'cfgrib'"
============================================================

That error means cfgrib is not installed/available in the Streamlit runtime.

Checklist:
1) `requirements.txt` MUST be at repo root (same level as SkyPulse/)
2) `packages.txt` MUST be at repo root (same level as SkyPulse/)
3) Push a commit that changes those files
4) In Streamlit Cloud: Manage app -> Reboot (or Clear cache + Reboot)
5) Open MRMS tab -> expand "MRMS decode diagnostics"
   - If cfgrib import fails, Streamlit did not reinstall deps.
   - If cfgrib OK but eccodes fails, OS packages aren't installed.
