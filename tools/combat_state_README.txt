HollowRL combat-state Mod update

This package updates only HollowRL.dll. It does not modify save slot 1, nail
damage, charms, abilities, difficulty, policy weights or Python installations.
It supplies the additional dash-state telemetry required by the new controller.

B machine:
1. Stop any trainer/controller connected to these games. Keep the original
   9003 and 9004 game windows running so their paths/arguments can be recovered.
2. Extract this ZIP to a normal writable folder, not inside the ZIP viewer.
3. Open PowerShell in that folder and run:
   powershell -ExecutionPolicy Bypass -File .\update_combat_state.ps1
4. The script checks the DLL hash and game identities, backs up each installed
   DLL, closes only the selected games, installs/verifies the Mod and restarts
   those games using their original executable and arguments.
5. Load save slot 1 in both windows and stand at the Traitor Lord statue.
   Do not enter the fight or select difficulty manually.
6. Tell the A-machine operator that both updated windows are ready. Training
   starts only after the controller verifies the new telemetry on all windows.

To inspect without closing/replacing/restarting anything:
   powershell -ExecutionPolicy Bypass -File .\update_combat_state.ps1 -CheckOnly

A machine uses the same updater with the intended ports:
   .\update_combat_state.ps1 -Ports 9001,9002
For a built DLL outside the package, add -DllPath 'C:\path\HollowRL.dll'.

Backups, original arguments, old/new hashes and installation results are saved
in combat_state_backups/<timestamp> beside this script. Keep the backup folder.
If verification or rollback fails, the script leaves the games closed; inspect
the printed error and backup result.json before retrying. It never starts a
game with an unverified installed DLL and never restarts unrelated windows.
