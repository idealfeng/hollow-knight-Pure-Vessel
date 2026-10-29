param(
    [string]$TypePattern = '^BossSceneController$',
    [string]$MethodPattern = '.',
    [string]$FieldPattern = '.',
    [switch]$IL,
    [string]$GameDir = 'D:\Program Files\Steam\steamapps\common\Hollow Knight'
)
$ErrorActionPreference = 'Stop'
[void][Reflection.Assembly]::Load([IO.File]::ReadAllBytes((Join-Path $GameDir 'BepInEx\core\Mono.Cecil.dll')))
$assembly = [Mono.Cecil.AssemblyDefinition]::ReadAssembly((Join-Path $GameDir 'hollow_knight_Data\Managed\Assembly-CSharp.dll'))
function Read-Types($types) {
    foreach ($type in $types) {
        if ($type.FullName -match $TypePattern) {
            Write-Output ('TYPE ' + $type.FullName)
            foreach ($field in $type.Fields) {
                if ($field.Name -match $FieldPattern) { Write-Output ('FIELD ' + $field.FullName) }
            }
            foreach ($method in $type.Methods) {
                if ($method.Name -match $MethodPattern) {
                    Write-Output ('METHOD ' + $method.FullName)
                    if ($IL -and $method.HasBody) {
                        foreach ($instruction in $method.Body.Instructions) { Write-Output $instruction.ToString() }
                    }
                }
            }
        }
        Read-Types $type.NestedTypes
    }
}
try { Read-Types $assembly.MainModule.Types } finally { $assembly.Dispose() }
